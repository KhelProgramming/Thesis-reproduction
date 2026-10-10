"""Keep box/mask namespaces and map compact metric arrays to their actual IDs."""
from .common import clean


def harmonic(p, r):
    return 2 * p * r / (p + r) if p + r else 0.0


def metric_block(metric, names):
    ids = [int(i) for i in metric.ap_class_index]
    positions = {class_id: row for row, class_id in enumerate(ids)}
    per_class = []
    for class_id, name in enumerate(names):
        row = dict(class_id=class_id, class_name=name, evaluated=class_id in positions,
                   precision=None, recall=None, f1=None, ap50=None, ap50_95=None)
        if class_id in positions:
            j = positions[class_id]
            row.update(precision=float(metric.p[j]), recall=float(metric.r[j]),
                       f1=float(metric.f1[j]), ap50=float(metric.ap50[j]), ap50_95=float(metric.ap[j]))
        per_class.append(row)
    p, r = float(metric.mp), float(metric.mr)
    return dict(precision=p, recall=r, f1_macro=sum(float(v) for v in metric.f1) / len(metric.f1) if len(metric.f1) else 0.0,
                f1_from_mean_precision_recall=harmonic(p, r), map50=float(metric.map50),
                map50_95=float(metric.map), evaluated_class_ids=ids, per_class=per_class)


def collect_metrics(metrics, names):
    return clean(dict(box=metric_block(metrics.box, names), mask=metric_block(metrics.seg, names),
                      ultralytics_results=metrics.results_dict, speed_ms_per_image=metrics.speed,
                      definitions={"f1_macro": "Mean of per-class Ultralytics F1 values over evaluated classes.",
                                   "f1_from_mean_precision_recall": "Harmonic mean of reported mean P and mean R; not generally equal to macro-F1.",
                                   "precision_recall": "Ultralytics validation operating point, not Android app confidence thresholds.",
                                   "iou": "No standalone IoU reproduction: the paper's aggregation/matching implementation is not supplied.",
                                   "missing_class": "null metrics mean not present in the compact AP class index; not filled with a fabricated score."}))
