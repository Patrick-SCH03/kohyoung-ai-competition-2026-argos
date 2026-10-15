"""Self-verification with the checker's rules and tolerances (plus fiducial exclusivity/index
order) and a conservative repair."""
import numpy as np
try:
    from .geometry import FOV
except ImportError:
    from geometry import FOV

CTOL = 1e-5   # checker tolerance


def _union_covers(rects, target, tol=CTOL):
    """Port of Checker.is_target_fully_covered."""
    tx1, ty1, tx2, ty2 = target
    filtered = []
    x_edges = {tx1, tx2}
    y_edges = {ty1, ty2}
    for x1, y1, x2, y2 in rects:
        if x2 <= tx1 - tol or x1 >= tx2 + tol or y2 <= ty1 - tol or y1 >= ty2 + tol:
            continue
        cx1, cy1 = max(x1, tx1), max(y1, ty1)
        cx2, cy2 = min(x2, tx2), min(y2, ty2)
        filtered.append((cx1, cy1, cx2, cy2))
        x_edges.update((cx1, cx2))
        y_edges.update((cy1, cy2))

    def dedup(edges):
        s = sorted(edges)
        out = [s[0]]
        for v in s[1:]:
            if v - out[-1] > tol:
                out.append(v)
        return out

    xl, yl = dedup(x_edges), dedup(y_edges)
    for i in range(len(xl) - 1):
        for j in range(len(yl) - 1):
            cx1, cx2, cy1, cy2 = xl[i], xl[i + 1], yl[j], yl[j + 1]
            if not (tx1 <= cx1 + tol and cx2 <= tx2 + tol and ty1 <= cy1 + tol and cy2 <= ty2 + tol):
                continue
            if not any(rx1 <= cx1 + tol and rx2 >= cx2 - tol and ry1 <= cy1 + tol and ry2 >= cy2 - tol
                       for rx1, ry1, rx2, ry2 in filtered):
                return False
    return True


def verify(P, fovs, seq):
    """Return list of issues: ('uncovered', comp) / ('outside', comp, fov) / ('mixed_step', fov)
    / ('order', pos) / ('fid_not_alone', fov) / ('fid_order', pos) / ('dup', comp)."""
    issues = []
    listed = np.zeros(P.n, dtype=int)
    big_rects = {int(i): [] for i in np.where(P.big)[0]}
    for k in seq:
        f = fovs[k]
        tlx, tly = f.x - P.fw / 2, f.y - P.fh / 2
        brx, bry = f.x + P.fw / 2, f.y + P.fh / 2
        stlx, stly = f.x - P.fw / 4, f.y - P.fh / 4
        sbrx, sbry = f.x + P.fw / 4, f.y + P.fh / 4
        steps = set()
        for m in f.members:
            listed[m] += 1
            steps.add(int(P.step[m]))
            if P.big[m]:
                if P.side[m] == 1:
                    big_rects[m].append((stlx, stly, sbrx, sbry))
                else:
                    big_rects[m].append((tlx, tly, brx, bry))
                continue
            if P.side[m] == 1:
                ok = (stlx - CTOL <= P.tlx[m] and stly - CTOL <= P.tly[m] and
                      sbrx + CTOL >= P.brx[m] and sbry + CTOL >= P.bry[m])
            else:
                ok = (tlx - CTOL <= P.tlx[m] and tly - CTOL <= P.tly[m] and
                      brx + CTOL >= P.brx[m] and bry + CTOL >= P.bry[m])
            if not ok:
                issues.append(('outside', int(m), int(k)))
        if len(steps) > 1:
            issues.append(('mixed_step', int(k)))
    for m in range(P.n):
        if listed[m] == 0:
            issues.append(('uncovered', int(m)))
        elif listed[m] > 1 and not P.big[m]:
            issues.append(('dup', int(m)))
    for m, rects in big_rects.items():
        if rects and not _union_covers(rects, (P.tlx[m], P.tly[m], P.brx[m], P.bry[m])):
            issues.append(('uncovered', int(m)))
    # order
    types = [fovs[k].type_of(P) for k in seq]
    for p in range(len(types) - 1):
        if types[p] < types[p + 1]:
            issues.append(('order', p))
    last_fid = -1
    for p, k in enumerate(seq):
        f = fovs[k]
        fids = [m for m in f.members if P.type[m] == 2]
        if fids:
            if len(f.members) != 1:
                issues.append(('fid_not_alone', int(k)))
            elif abs(f.x - P.cx[fids[0]]) > 1e-9 or abs(f.y - P.cy[fids[0]]) > 1e-9:
                issues.append(('fid_off_center', int(k)))
            if min(fids) < last_fid:
                issues.append(('fid_order', p))
            last_fid = max(last_fid, max(fids))
    return issues


def repair(P, fovs, seq, issues):
    """Conservative fix: remove offending memberships and add dedicated FOVs."""
    fovs = list(fovs)
    seq = list(seq)
    to_add = set()
    for it in issues:
        if it[0] == 'outside':
            m, k = it[1], it[2]
            if m in fovs[k].members:
                fovs[k].members.remove(m)
            to_add.add(m)
        elif it[0] == 'uncovered':
            to_add.add(it[1])
        elif it[0] == 'dup':
            m = it[1]
            owners = [k for k in seq if m in fovs[k].members]
            for k in owners[1:]:
                fovs[k].members.remove(m)
        elif it[0] == 'mixed_step':
            k = it[1]
            f = fovs[k]
            s0 = P.step[f.members[0]]
            bad = [m for m in f.members if P.step[m] != s0]
            for m in bad:
                f.members.remove(m)
                to_add.add(m)
        elif it[0] == 'fid_not_alone':
            k = it[1]
            f = fovs[k]
            others = [m for m in f.members if P.type[m] != 2]
            for m in others:
                f.members.remove(m)
                to_add.add(m)
        elif it[0] == 'fid_off_center':
            f = fovs[it[1]]
            f.x, f.y = float(P.cx[f.members[0]]), float(P.cy[f.members[0]])
    new = []
    for m in sorted(to_add):
        if P.big[m]:
            for (x, y) in P.tiles(m):
                new.append(FOV(x, y, [m], P.step[m], fixed=True))
        else:
            new.append(FOV(P.cx[m], P.cy[m], [m], P.step[m], exclusive=(P.type[m] == 2)))
    base = len(fovs)
    fovs.extend(new)
    new_ids = list(range(base, base + len(new)))
    # drop emptied FOVs from the sequence
    seq = [k for k in seq if fovs[k].members]
    # insert new FOVs by type block (append at the end of their type block)
    for k in new_ids:
        t = fovs[k].type_of(P)
        pos = len(seq)
        for p in range(len(seq) - 1, -1, -1):
            if fovs[seq[p]].type_of(P) >= t:
                pos = p + 1
                break
        else:
            pos = 0
        seq.insert(pos, k)
    return fovs, seq
