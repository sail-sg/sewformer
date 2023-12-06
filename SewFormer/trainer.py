# Training loop func
from pathlib import Path
import time
import traceback
import shutil
import os


import torch
import wandb as wb
import torchvision.transforms as T

# My modules
import data
from data.transforms import denormalize_img_transforms
# from data.transforms import make_image_augments
from warm_cosine_scheduler import GradualWarmupScheduler

class Trainer():
    def __init__(
            self, 
            setup, experiment_tracker, dataset=None, data_split={}, 
            with_norm=True, with_visualization=False):
        """Initialize training and dataset split (if given)
            * with_visualization toggles image prediction logging to wandb board. Only works on custom garment datasets (with prediction -> image) conversion"""


        self.experiment = experiment_tracker
        self.datawraper = None
        self.standardize_data = with_norm
        self.log_with_visualization = with_visualization
        
        # training setup
        self.setup = setup

        if dataset is not None:
            self.use_dataset(dataset, data_split)
    
    def init_randomizer(self, random_seed=None):
        """Init randomizatoin for torch globally for reproducibility. 
            Using this function ensures that random seed will be recorded in config
        """
        # see https://pytorch.org/docs/stable/notes/randomness.html
        if random_seed:
            self.setup['random_seed'] = random_seed
        elif not self.setup['random_seed']:
            self.setup['random_seed'] = int(time.time())

        torch.manual_seed(self.setup['random_seed'])
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    def use_dataset(self, dataset, split_info):
        """Use specified dataset for training with given split settings"""
        exp_config = self.experiment.in_config
        if 'wrapper' in exp_config["dataset"] and exp_config["dataset"]["wrapper"] is not None:
            datawrapper_class = getattr(data, exp_config["dataset"]["wrapper"])
            print("datawrapper_class", datawrapper_class)
            self.datawraper = datawrapper_class(dataset)
        else:
            self.datawraper = data.RealisticDatasetDetrWrapper(dataset)
        self.datawraper.load_split(split_info)
        self.datawraper.new_loaders(self.setup['batch_size'], shuffle_train=True)

        if self.standardize_data:
            self.datawraper.standardize_data()
        return self.datawraper
    
    def fit(self, model, config=None):
        """Fit provided model to reviosly configured dataset"""
        if not self.datawraper:
            raise RuntimeError('Trainer::Error::fit before dataset was provided. run use_dataset() first')

        self.device = model.device_ids if hasattr(model, 'device_ids') and len(model.device_ids) > 0 else self.setup['devices']
        
        self._add_optimizer(model)
        self._add_scheduler(len(self.datawraper.loaders.train))
        self.es_tracking = []  # early stopping init
        start_epoch = self._start_experiment(model, config=config)
        print('{}::NN training Using device: {}'.format(self.__class__.__name__, self.device))
        time.sleep(10)

        if self.log_with_visualization:
            # to run parent dir -- wandb will automatically keep track of intermediate values
            # Othervise it might only display the last value (if saving with the same name every time)
            self.folder_for_preds = Path('./wandb') / 'intermediate_preds_{}_folder'.format(self.__class__.__name__)
            self.folder_for_preds.mkdir(exist_ok=True)
        
        self._fit_loop(model, self.datawraper.loaders.train, self.datawraper.loaders.validation, start_epoch=start_epoch)

        print("{}::Finished training".format(self.__class__.__name__))
        # self.experiment.stop() -- not stopping the run for convenice for further processing outside of the training routines

    # ---- Private -----
    def _fit_loop(self, model, train_loader, valid_loader, start_epoch=0):
        """Fit loop with the setup already performed. Assumes wandb experiment was initialized"""

        log_step = wb.run.step - 1
        best_valid_loss = self.experiment.last_best_validation_loss()
        best_valid_loss = torch.tensor(best_valid_loss) if best_valid_loss is not None else None
        
        for epoch in range(start_epoch, wb.config.trainer['epochs']):
            model.train()
            for i, batch in enumerate(train_loader):
                features, gt = batch['features'].to(self.device[0]), batch['ground_truth']   # .to(self.device)
                
                # with torch.autograd.detect_anomaly():
                loss, loss_dict, loss_structure_update = model.module.loss(model(features, log_step=log_step, epoch=epoch), gt, epoch=epoch)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                if self.scheduler is not None:
                    self.scheduler.step()
                if hasattr(model.module, 'step'):  # custom model hyperparams scheduling
                    model.module.step(i, len(train_loader))
                
                # logging
                log_step += 1
                loss_dict.update({'epoch': epoch, 'batch': i, 'loss': loss, 'learning_rate': self.optimizer.param_groups[0]['lr']})
                wb.log(loss_dict, step=log_step)

            # Check the cluster assignment history
            if hasattr(model.module.loss, 'cluster_resolution_mapping') and model.module.loss.debug_prints:
                print(model.module.loss.cluster_resolution_mapping)

            # scheduler step: after optimizer step, see https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
            model.eval()
            with torch.no_grad():
                losses = [model.module.loss(model(batch['features'].to(self.device[0])), batch['ground_truth'], epoch=epoch)[0] for batch in valid_loader]
                valid_loss = sum(losses) / len(losses)  # Each loss element is already a mean for its batch

            # Checkpoints: & compare with previous best
            if loss_structure_update or best_valid_loss is None or valid_loss < best_valid_loss:  # taking advantage of lazy evaluation
                best_valid_loss = valid_loss
                self._save_checkpoint(model, epoch, best=True)  # saving only the good models
            else:
                self._save_checkpoint(model, epoch)

            # Base logging
            print('Epoch: {}, Validation Loss: {}'.format(epoch, valid_loss))
            wb.log({'epoch': epoch, 'valid_loss': valid_loss, 'best_valid_loss': best_valid_loss}, step=log_step)

            # prediction for visual reference
            if self.log_with_visualization:
                self._log_an_image(model, valid_loader, epoch, log_step)

            # check for early stoping
            if self._early_stopping(loss, best_valid_loss, self.optimizer.param_groups[0]['lr']):
                print('{}::Stopped training early'.format(self.__class__.__name__))
                break
    
    def _start_experiment(self, model, config=None):
        self.experiment.init_run({'trainer': self.setup})

        if wb.run.resumed:
            start_epoch = self._restore_run(model)
            self.experiment.checkpoint_counter = start_epoch
            print('{}::Resumed run {} from epoch {}'.format(self.__class__.__name__, self.experiment.cloud_path(), start_epoch))

            if self.device != wb.config.trainer['devices'][0]:
                # device doesn't matter much, so we just inform but do not crash
                print('{}::Warning::Resuming run on different device. Was {}, now using {}'.format(
                    self.__class__.__name__, wb.config.trainer['devices'][0], self.device))
        elif config is not None and config["NN"]["pre-trained"] is not None:
            checkpoint = torch.load(config["NN"]["pre-trained"], map_location="cpu")
            # checkpoint loaded correctly
            model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if self.scheduler is not None:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])  # https://discuss.pytorch.org/t/how-to-save-and-load-lr-scheduler-stats-in-pytorch/20208

            # new epoch id
            start_epoch = checkpoint['epoch'] + 1
            print('{}::Resumed run {} from epoch {}'.format(self.__class__.__name__, config["NN"]["step-trained"], start_epoch))

        else:
            start_epoch = 0
            # record configurations of data and model
            self.datawraper.save_to_wandb(self.experiment)
            if hasattr(model.module, 'config'):
                self.experiment.add_config('NN', model.module.config)  # save NN configuration
            elif config is not None:
                self.experiment.add_config('NN', config["NN"])

        wb.watch(model, log='all')
        return start_epoch
    

    def _add_optimizer(self, model):
        if self.setup['optimizer'] == 'SGD':
            # future 'else'
            print('{}::Using default SGD optimizer'.format(self.__class__.__name__))
            self.optimizer = torch.optim.SGD(model.parameters(), lr=self.setup['learning_rate'], weight_decay=self.setup['weight_decay'])
        elif self.setup['optimizer'] == 'Adam':
            # future 'else'
            print('{}::Using Adam optimizer'.format(self.__class__.__name__))
            model.to(self.device[0])  # see https://discuss.pytorch.org/t/effect-of-calling-model-cuda-after-constructing-an-optimizer/15165/8
            self.optimizer = torch.optim.Adam(model.parameters(), lr=self.setup['learning_rate'], weight_decay=self.setup['weight_decay'])

    def _add_scheduler(self, steps_per_epoch):
        if 'lr_scheduling' in self.setup:
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer, 
                max_lr=self.setup['learning_rate'],
                epochs=self.setup['epochs'],
                steps_per_epoch=steps_per_epoch,
                cycle_momentum=False  # to work with Adam
            )
        else:
            self.scheduler = None
            print('{}::Warning::no learning scheduling set'.format(self.__class__.__name__))

    def _restore_run(self, model):
        """Restore the training process from the point it stopped at. 
            Assuming 
                * Current wb.config state is the same as it was when run was initially created
                * All the necessary training objects are already created and only need update
                * All related object types are the same as in the resuming run (model, optimizer, etc.)
                * Self.run_id is properly set
            Returns id of the next epoch to resume from. """
        
        # data split
        split, batch_size, data_config = self.experiment.data_info()

        self.datawraper.dataset.update_config(data_config)
        self.datawraper.load_split(split, batch_size)  # NOTE : random number generator reset

        # get latest checkoint info
        print('{}::Loading checkpoint to resume run..'.format(self.__class__.__name__))
        checkpoint = self.experiment.get_checkpoint_file()  # latest

        # checkpoint loaded correctly
        model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])  # https://discuss.pytorch.org/t/how-to-save-and-load-lr-scheduler-stats-in-pytorch/20208

        # new epoch id
        return checkpoint['epoch'] + 1
    
    def _early_stopping(self, last_loss, last_tracking_loss, last_lr):
        """Check if conditions are met to stop training. Returns a message with a reason if met
            Early stopping allows to save compute time"""

        # loss goes into nans
        if torch.isnan(last_loss):
            self.experiment.add_statistic('stopped early', 'Nan in losses', log='{}::EarlyStopping'.format(self.__class__.__name__))
            return True

        # Target metric is not improving for some time
        self.es_tracking.append(last_tracking_loss.item())
        if len(self.es_tracking) > (wb.config.trainer['early_stopping']['patience'] + 1):  # number of last calls to consider plus current -> at least two
            self.es_tracking.pop(0)
            # if all values fit into a window, they don't change much
            if abs(max(self.es_tracking) - min(self.es_tracking)) < wb.config.trainer['early_stopping']['window']:
                self.experiment.add_statistic(
                    'stopped early', 'Metric have not changed for {} epochs'.format(wb.config.trainer['early_stopping']['patience']), 
                    log='Trainer::EarlyStopping')
                return True
        # do not check untill wb.config.trainer['early_stopping'].patience # of calls are gathered

        # Learning rate vanished
        if last_lr < 1e-8:
            self.experiment.add_statistic('stopped early', 'Learning Rate vanished', log='Trainer::EarlyStopping')
            return True
        
        return False

    def _log_an_image(self, model, loader, epoch, log_step):
        """Log image of one example prediction to wandb.
            If the loader does not shuffle batches, logged image is the same on every step"""
        with torch.no_grad():
            # using one-sample-from-each-of-the-base-folders loader
            single_sample_loader = self.datawraper.loaders.valid_single_per_data
            if single_sample_loader is None:
                print('{}::Error::Suitable loader is not available. Nothing logged'.format(self.__class__.__name__))

            try: 
                img_files = []
                for batch in single_sample_loader:

                    batch_img_files = self.datawraper.dataset.save_prediction_batch(
                        model(batch['features'].to(self.device)), 
                        batch['img_fn'], batch['data_folder'],
                        save_to=self.folder_for_preds, images=None)

                    img_files += batch_img_files
            except BaseException as e:
                print(e)
                traceback.print_exc()
                print('{}::Error::On saving pattern prediction for image logging. Nothing logged'.format(self.__class__.__name__))
            else:
                for i in range(len(img_files)):
                    print('{}::Logged pattern prediction for {}'.format(self.__class__.__name__, img_files[i].name))
                    try:
                        wb.log({img_files[i].name: [wb.Image(str(img_files[i]))], 'epoch': epoch}, step=log_step)  # will raise errors if given file is not an image
                    except BaseException as e:
                        print(e)
                        pass

    def _save_checkpoint(self, model, epoch, best=False):
        """Save checkpoint that can be used to resume training"""
        
        # https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-a-general-checkpoint-for-inference-and-or-resuming-training
        checkpoint_dict = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }
        if self.scheduler is not None:
            checkpoint_dict['scheduler_state_dict'] = self.scheduler.state_dict()

        self.experiment.save_checkpoint(
            checkpoint_dict,
            aliases=['best'] if best else [], 
            wait_for_upload=best
        )

class TrainerDetr(Trainer):
    def __init__(self,
                 setup, experiment_tracker, dataset=None, data_split={}, 
                 with_norm=True, with_visualization=False):
        super().__init__(setup, experiment_tracker, dataset=dataset, data_split=data_split, 
                         with_norm=with_norm, with_visualization=with_visualization)
        self.denorimalize = denormalize_img_transforms()
    
    def _add_optimizer(self, model_without_ddp):
        param_dicts = [
            {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": float(self.setup["lr_backbone"]),
            },
        ]

        self.optimizer = torch.optim.AdamW(param_dicts, lr=float(self.setup["lr"] / 8),
                                    weight_decay=float(self.setup["weight_decay"]))
        print('TrainerDetr::Using AdamW optimizer')
    
    def _add_scheduler(self, steps_per_epoch):
        if 'lr_scheduling' in self.setup and self.setup["lr_scheduling"] == "OneCycleLR":
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer, 
                max_lr=self.setup['lr'],
                epochs=self.setup['epochs'],
                steps_per_epoch=steps_per_epoch,
                cycle_momentum=False  # to work with Adam
            )
        elif 'lr_scheduling' in self.setup and self.setup["lr_scheduling"] == "warm_cosine":

            consine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, 
                                                                           T_max=self.setup["epochs"] * steps_per_epoch, 
                                                                           eta_min=0, 
                                                                           last_epoch=-1)
            self.scheduler = GradualWarmupScheduler(self.optimizer, multiplier=8, total_epoch=5 * steps_per_epoch, after_scheduler=consine_scheduler)

        else:
            self.scheduler = None 
            print('TrainerDetr::Warning::no learning scheduling set')
    
    def use_dataset(self, dataset, split_info):
        """Use specified dataset for training with given split settings"""
        exp_config = self.experiment.in_config
        if 'wrapper' in exp_config["dataset"] and exp_config["dataset"]["wrapper"] is not None:
            datawrapper_class = getattr(data, exp_config["dataset"]["wrapper"])
            print("datawrapper_class", datawrapper_class)
            self.datawraper = datawrapper_class(dataset)
        else:
            self.datawraper = data.RealisticDatasetDetrWrapper(dataset)
        
        # self.datawraper = data.RealisticDatasetDetrWrapper(dataset)
        self.datawraper.load_split(split_info)
        self.datawraper.new_loaders(self.setup['batch_size'], shuffle_train=True, multiprocess=self.setup["multiprocess"])

        if self.standardize_data:
            self.datawraper.standardize_data()

        return self.datawraper
    
    def fit(self, model, model_without_ddp, criterion, rank=0, config=None):
        """Fit provided model to reviosly configured dataset"""

        if not self.datawraper:
            raise RuntimeError('{}::Error::fit before dataset was provided. run use_dataset() first'.format(self.__class__.__name__))
        if self.setup["multiprocess"]:
            self.device = rank
        else:
            self.device = ["cuda:{}".format(did) for did in model.device_ids] if hasattr(model, 'device_ids') \
                                           and len(model.device_ids) > 0 else self.setup['devices']
            self.device = 'cpu' if len(self.device) == 0 else self.device[0]
        
        self._add_optimizer(model_without_ddp)
        self._add_scheduler(len(self.datawraper.loaders.train))
        self.es_tracking = []  # early stopping init
        start_epoch = self._start_experiment(model, config)
        print('{}::NN training Using device: {}'.format(self.__class__.__name__, self.device))
        
        if self.log_with_visualization:
            # to run parent dir -- wandb will automatically keep track of intermediate values
            # Othervise it might only display the last value (if saving with the same name every time)
            self.folder_for_preds = Path('./wandb') / 'intermediate_preds_{}'.format(self.__class__.__name__)
            self.folder_for_preds.mkdir(exist_ok=True)
        
        self._fit_loop_without_matcher(model, criterion, self.datawraper.loaders.train, self.datawraper.loaders.validation, start_epoch=start_epoch)
        print("{}::Finished training".format(self.__class__.__name__))
    

    def _fit_loop_without_matcher(self, model, criterion, train_loader, valid_loader, start_epoch):
        """Fit loop with the setup already performed. Assumes wandb experiment was initialized"""

        global best_valid_loss

        # self.setup["dry_run"] = True
        log_step = wb.run.step - 1
        return_stitches = self.setup["return_stitches"]

        if (self.setup["multiprocess"] and self.device == 0) or not self.setup["multiprocess"]:
            best_valid_loss = self.experiment.last_best_validation_loss()
            best_valid_loss = torch.tensor(best_valid_loss) if best_valid_loss is not None else None
        iter_items = 0
        for epoch in range(start_epoch, wb.config.trainer["epochs"]):
            model.train()
            criterion.train()
            self.datawraper.dataset.set_training(True)
            for i, batch in enumerate(train_loader):
                iter_items += 1
                images, gt = batch['image'], batch['ground_truth']
                images = images.to(self.device)
                outputs = model(images,
                                gt_stitches=gt["masked_stitches"], 
                                gt_edge_mask=gt["stitch_edge_mask"], 
                                return_stitches=return_stitches)
                loss, loss_dict = criterion(outputs, gt, epoch=epoch)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if self.scheduler is not None:
                    self.scheduler.step()
                
                log_step += 1
                loss_dict.update({'epoch': epoch, 
                                  'batch': i, 
                                  'loss': loss, 
                                  'learning_rate': self.optimizer.param_groups[0]['lr']})
                wb.log(loss_dict, step=log_step)
                if iter_items % 10 == 0:
                    print(f"epoch: {epoch:02d}, batch: {i:04d}, loss: {loss:.6f}, lr: {self.optimizer.param_groups[0]['lr']:.6f}")
                
                if (self.setup["multiprocess"] and self.device == 0) or not self.setup["multiprocess"]:
                    if self.setup["dry_run"]:
                        break
            
            model.eval()
            criterion.eval()
            self.datawraper.dataset.set_training(False)
            with torch.no_grad():
                valid_losses, valid_loss_dict = [], {}
                for batch in valid_loader:
                    images, gt = batch['image'], batch['ground_truth']
                    images = images.to(self.device)
                    outputs = model(images, 
                                    gt_stitches=gt["masked_stitches"], 
                                    gt_edge_mask=gt["stitch_edge_mask"], 
                                    return_stitches=return_stitches)
                    loss, loss_dict = criterion(outputs, gt, epoch=epoch)
                    valid_losses.append(loss)
                    if len(valid_loss_dict) == 0:
                        valid_loss_dict = {'valid_' + key: [] for key in loss_dict}
                    for key, val in loss_dict.items():
                        if val is not None:
                            valid_loss_dict['valid_' + key].append(val)
                    
                    if (self.setup["multiprocess"] and self.device == 0) or not self.setup["multiprocess"]:
                        if self.setup["dry_run"]:
                            break
                valid_loss = sum(valid_losses) / len(valid_losses)  # Each loss element is already a mean for its batch
                valid_loss_dict = {key: sum(val)/len(val) if len(val) > 0 else None for key, val in valid_loss_dict.items()}

            # Checkpoints: & compare with previous best
            if (self.setup["multiprocess"] and self.device == 0) or not self.setup["multiprocess"]:
                if best_valid_loss is None or valid_loss < best_valid_loss:  # taking advantage of lazy evaluation
                    best_valid_loss = valid_loss
                    self._save_checkpoint(model, epoch, best=True)  # saving only the good models

                else:
                    self._save_checkpoint(model, epoch)

                # Base logging
                print('!!!!!!!!!!!!!!!!!!!!!!!!!!!Epoch: {}, Validation Loss: {}'.format(epoch, valid_loss))
                valid_loss_dict.update({'epoch': epoch, 'valid_loss': valid_loss, 'best_valid_loss': best_valid_loss})
                wb.log(valid_loss_dict, step=log_step)

                # prediction for visual reference
                if self.log_with_visualization:
                    self.datawraper.dataset.set_training(False)
                    valid_batch = iter(valid_loader).__next__()
                    self._log_batch_image(model, valid_batch, epoch, log_step, tag="valid", return_stitches=return_stitches)
                 
    def _log_batch_image(self, model, batch_sample, epoch, log_step, tag="valid", return_stitches=False):
        with torch.no_grad():
            try:
                batch_size = 1
                image = batch_sample["image"][:batch_size]
                gt = {key:val[:batch_size] for key, val in batch_sample["ground_truth"].items()}
                name, folder, img_fn = batch_sample["name"][:batch_size], batch_sample["data_folder"][:batch_size], batch_sample["img_fn"][:batch_size]
                inputs = image.to(self.device)

                batch_img_files = self.datawraper.dataset.save_prediction_batch(
                    model(image.to(self.device), gt_stitches=gt["masked_stitches"], gt_edge_mask=gt["stitch_edge_mask"], return_stitches=False), 
                    img_fn, 
                    folder, 
                    save_to=self.folder_for_preds,
                    images=image)
                batch_gt_files = self.datawraper.dataset.save_gt_batch_imgs(
                        gt, img_fn, folder, save_to=self.folder_for_preds
                    )
            except BaseException as e:
                print(e)
                traceback.print_exc()
                print('{}::Error::On saving pattern prediction for image logging. Nothing logged'.format(self.__class__.__name__))
            else:
                print('{}::Logged pattern prediction for {}'.format(self.__class__.__name__, batch_gt_files[0].name))
                try:
                    wb.log({"Input:#{}".format(tag + str(0)): [wb.Image(T.ToPILImage()(batch_sample["image"][0].cpu()))], 'epoch':epoch}, step=log_step)
                    wb.log({"GT: #{}".format(tag + str(0)): [wb.Image(str(batch_gt_files[0]))], 'epoch': epoch}, step=log_step)
                    wb.log({"Output:#{}".format(tag + str(0)): [wb.Image(str(batch_img_files[0]))], 'epoch': epoch}, step=log_step) 
                except BaseException as e:
                        print(e)
                        pass
                

        
