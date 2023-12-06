import torch
import time

# My modules
from data import InvalidPatternDefError

def eval_detr_metrics(model, criterion, data_warpper, rank=0, section='test'):

    device = 'cuda:{}'.format(rank) if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()
    criterion.to(device)
    criterion.eval()
    criterion.with_quality_eval()
    data_warpper.dataset.set_training(False)

    with torch.no_grad():
        loader = data_warpper.get_loader(data_section=section)
        print("eval on {}, len = {}".format(section, len(loader)))
        return _eval_detr_metrics_per_loader(model, criterion, loader, device)


def _eval_detr_metrics_per_loader(model, criterion, loader, device):
    current_metrics = dict.fromkeys(['full_loss'], [])
    counter = 0
    loader_iter = iter(loader)
    start_time = time.time()

    score_dict = {}
    collect_keys = ["st_f1s"]
    best_score_dict = {}

    while True:
        try:
            batch = next(loader_iter)
        except StopIteration:
            break
        except InvalidPatternDefError as e:
            print(e)
            continue
        images, gt = batch["image"].to(device), batch["ground_truth"]
        img_name = batch["img_fn"]

        outputs = model(images)
        full_loss, loss_dict = criterion(outputs, gt, epoch = -1)
        
        # gathering up
        current_metrics['full_loss'].append(full_loss.cpu().numpy())
        for key, value in loss_dict.items():
            if key not in current_metrics:
                current_metrics[key] = []  # init new metric
            if value is not None:  # otherwise skip this one from accounting for!
                value = value.cpu().numpy() if isinstance(value, torch.Tensor) else value
                if isinstance(value, list):
                    current_metrics[key].extend(value)
                else:
                    current_metrics[key].append(value)
        counter += 1
        if counter % 10 == 0:
            print("eval progress: {}, time cost={:.2f}".format(counter, time.time() - start_time))
    print(f"Total eval batches: {counter}, time cost={time.time() - start_time:.2f}")

    # sum & normalize 
    for metric in current_metrics:
        if len(current_metrics[metric]):
            current_metrics[metric] = sum(current_metrics[metric]) / len(current_metrics[metric])
        else:
            current_metrics[metric] = None

    return current_metrics, score_dict, best_score_dict

# ----- Utils -----
def eval_pad_vector(data_stats={}):
    # prepare padding vector used for panel padding 
    if data_stats:
        shift = torch.Tensor(data_stats['shift'])
        scale = torch.Tensor(data_stats['scale'])
        return (- shift / scale)
    else:
        return None