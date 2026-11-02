"""Conditional score uncertainty for fixed predictions, keeping event blocks together."""
import numpy as np
import pandas as pd


def event_labels(frame):
    # Unnamed claims are pooled by year, not treated as independent storms.
    return frame.floodEvent.fillna('unnamed-' + frame.loss_year.astype(str)).astype(str)


def event_score_audit(y, predictions, groups, metric, pairs, repeats=1000, seed=42):
    """Paired whole-event bootstrap and deletion sensitivity; no model refitting.

    Each replicate draws K of the K observed event blocks with replacement.
    All candidates receive the same draws. Percentile bands are descriptive,
    conditional on these predictions and observed events, without selection correction.
    """
    y = np.asarray(y, dtype=float)
    codes, names = pd.factorize(np.asarray(groups), sort=True)
    k = len(names)
    if k < 3:
        raise ValueError('At least three event blocks are needed for this audit.')
    counts = np.bincount(codes, minlength=k)
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(k, np.full(k, 1/k), size=repeats)
    weight_sets = np.vstack([np.ones(k), draws, np.ones((k, k))-np.eye(k)])
    n = weight_sets @ counts
    values = {}
    if metric == 'R2':
        sums = np.bincount(codes, weights=y, minlength=k)
        squares = np.bincount(codes, weights=y*y, minlength=k)
        denominator = weight_sets @ squares - (weight_sets @ sums)**2/n
        for name, pred in predictions.items():
            errors = np.bincount(codes, weights=(y-np.asarray(pred))**2, minlength=k)
            values[name] = np.divide(weight_sets @ errors, denominator,
                                     out=np.full(len(n), np.nan), where=denominator>0)
            values[name] = 1-values[name]
    elif metric == 'PR-AUC':
        # Average precision with tied scores grouped exactly as sklearn does.
        # Pre-sort once; multiplicity weights equal physically repeated claims.
        for name, pred in predictions.items():
            pred = np.asarray(pred)
            order = np.argsort(-pred, kind='stable')
            ends = np.r_[np.flatnonzero(np.diff(pred[order])), len(y)-1]
            yy = y[order]; cc = codes[order]
            scores = []
            for weights in weight_sets:
                w = weights[cc]
                tp = np.cumsum(w*yy)[ends]
                total = np.cumsum(w)[ends]
                if tp[-1] == 0 or total[-1] == tp[-1]:
                    scores.append(np.nan)
                    continue
                precision = np.divide(tp,total,out=np.zeros_like(tp),where=total>0)
                scores.append(np.sum(np.diff(np.r_[0,tp])*precision)/tp[-1])
            values[name] = np.asarray(scores)
    else:
        raise ValueError(metric)
    summaries = []
    for name, v in values.items():
        boot = v[1:repeats+1]; deleted = v[repeats+1:]
        summaries.append({'candidate':name, metric:v[0],
                          'event bootstrap 2.5%':np.nanquantile(boot,.025),
                          'event bootstrap 97.5%':np.nanquantile(boot,.975),
                          'omit-event minimum':np.nanmin(deleted),
                          'omit-event maximum':np.nanmax(deleted),
                          'valid bootstrap draws':np.isfinite(boot).sum()})
    differences = []; deletion_rows = []
    for left,right in pairs:
        delta = values[left]-values[right]
        boot = delta[1:repeats+1]; deleted = delta[repeats+1:]
        differences.append({'comparison':left+' minus '+right,'difference':delta[0],
                            'paired 2.5%':np.nanquantile(boot,.025),
                            'paired 97.5%':np.nanquantile(boot,.975),
                            'omit-event minimum':np.nanmin(deleted),
                            'omit-event maximum':np.nanmax(deleted)})
        for event, value in zip(names, deleted):
            deletion_rows.append({'comparison':left+' minus '+right,
                                  'omitted event':event,'difference':value})
    blocks = pd.DataFrame({'event':names,'claims':counts}).set_index('event').sort_values('claims',ascending=False)
    return (pd.DataFrame(summaries).set_index('candidate'),
            pd.DataFrame(differences).set_index('comparison'),
            pd.DataFrame(deletion_rows), blocks)
