"""Inference masks for FusionTrack section IV-F (no Object Update Module)."""

import torch


def mutual_topk_edges(embeddings, cameras, top_k, threshold):
    """Select k candidates across ALL other views, then require reciprocity."""
    similarities = embeddings @ embeddings.T
    cameras = torch.as_tensor(cameras, device=embeddings.device)
    eligible = cameras[:, None] != cameras[None, :]
    similarities = similarities.masked_fill(~eligible, -torch.inf)
    k = min(top_k, len(cameras))
    selected = torch.zeros_like(eligible)
    if k:
        indices = similarities.argsort(dim=1, descending=True, stable=True)[:, :k]
        selected.scatter_(1, indices, True)
    mutual = selected & selected.T & eligible & (similarities >= threshold)
    return {(a, b): float(similarities[a, b])
            for a, b in mutual.triu(diagonal=1).nonzero().tolist()}


def neighbor_support(left, right, edges):
    """Fraction supported by distinct correspondences; cannot exceed one.

    The paper does not specify a denominator for unequal neighborhoods. Use
    the larger size so an unmatched neighborhood cannot inflate confidence.
    Empty neighborhoods provide no spatial evidence (caller decides policy).
    """
    matched = {}

    def augment(a, visited):
        for b in right:
            if b in visited or (a, b) not in edges and (b, a) not in edges:
                continue
            visited.add(b)
            if b not in matched or augment(matched[b], visited):
                matched[b] = a
                return True
        return False

    count = sum(augment(a, set()) for a in left)
    return count / max(len(left), len(right), 1)
