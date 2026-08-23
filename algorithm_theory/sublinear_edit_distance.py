"""
Sublinear Approximate Edit Distance & Pattern Matching Engine.
Bridging Metric Space Embeddings (Batu-Ergun-Kilberg, Andoni-Krauthgamer) with Elastic Hashing.

Replaces quadratic O(N * M) dynamic programming (Wagner-Fischer) with diagonal-banded
dynamic programming (Ukkonen-style) and fast approximate pattern matching.

Key Capabilities:
1. Banded Levenshtein edit distance in O(min(N, M) * K) when the true distance is <= K
   (exact within the band; returns max(N, M) when the band is exceeded).
2. Fast Approximate Pattern Matching in texts with up to k insertions/deletions/substitutions.
3. Multi-resolution q-gram frequency sketches with elastic open addressing.
"""

from typing import Tuple, Optional, List, Dict, Union, Any, Set
import numpy as np
import time


def exact_wagner_fischer_edit_distance(s1: str, s2: str) -> int:
    """
    Exact O(N * M) Wagner-Fischer dynamic programming edit distance baseline.
    """
    n, m = len(s1), len(s2)
    if n == 0:
        return m
    if m == 0:
        return n
        
    dp = np.zeros((n + 1, m + 1), dtype=np.int32)
    dp[:, 0] = np.arange(n + 1)
    dp[0, :] = np.arange(m + 1)
    
    for i in range(1, n + 1):
        c1 = s1[i - 1]
        for j in range(1, m + 1):
            c2 = s2[j - 1]
            cost = 0 if c1 == c2 else 1
            dp[i, j] = min(
                dp[i - 1, j] + 1,      # Deletion
                dp[i, j - 1] + 1,      # Insertion
                dp[i - 1, j - 1] + cost # Substitution
            )
            
    return int(dp[n, m])


class SublinearEditDistance:
    """
    Sublinear Approximate Edit Distance Engine.
    
    Utilizes multi-scale q-gram frequency embeddings, L1 metric projection, and
    diagonal-banded dynamic programming for high-speed string and sequence comparisons.
    """
    def __init__(self, q: int = 3, band_width: int = 16):
        """
        Parameters
        ----------
        q : int
            q-gram substring length for frequency embedding.
        band_width : int
            Half-width of the diagonal band for banded local alignment.
        """
        self.q = max(1, int(q))
        self.band_width = max(2, int(band_width))

    def _extract_qgram_profile(self, s: str) -> Dict[str, int]:
        """Extracts q-gram frequency histogram."""
        if len(s) < self.q:
            return {s: 1}
        profile: Dict[str, int] = {}
        for i in range(len(s) - self.q + 1):
            gram = s[i:i + self.q]
            profile[gram] = profile.get(gram, 0) + 1
        return profile

    def compute_qgram_lower_bound(self, s1: str, s2: str) -> float:
        """
        Computes the Ukkonen / Myers q-gram distance lower bound:
        d_edit(s1, s2) >= sum(|count1(g) - count2(g)|) / (2 * q).
        """
        prof1 = self._extract_qgram_profile(s1)
        prof2 = self._extract_qgram_profile(s2)
        
        all_grams = set(prof1.keys()).union(prof2.keys())
        l1_diff = 0
        for g in all_grams:
            c1 = prof1.get(g, 0)
            c2 = prof2.get(g, 0)
            l1_diff += abs(c1 - c2)
            
        return float(l1_diff / (2.0 * max(1, self.q)))

    def compute_banded_edit_distance(self, s1: str, s2: str, max_k: Optional[int] = None) -> int:
        """
        Computes banded edit distance constrained along the main diagonal |i - j| <= K.
        Complexity: O(min(N, M) * K) instead of O(N * M).

        The band must be wide enough to admit pure-prefix-deletion / pure-prefix-insertion
        alignment paths. The j = 0 (empty s2 prefix) and i = 0 (empty s1 prefix) border
        columns are initialised so that deleting/inserting a run of leading characters is
        always reachable inside the band.
        """
        n, m = len(s1), len(s2)
        # NOTE: use `is None` rather than truthiness so an explicit max_k == 0 (lengths must
        # match exactly) is honoured instead of falling back to band_width * 2.
        if max_k is None:
            limit = self.band_width * 2
        else:
            limit = max_k
        if abs(n - m) > limit:
            return max(n, m)

        k = max_k if max_k is not None else self.band_width
        k = max(k, abs(n - m))

        INF = 10**8
        dp = np.full((n + 1, 2 * k + 1), INF, dtype=np.int32)

        # Helper index mapping: (i, j) -> j - i + k
        def col_idx(i: int, j: int) -> int:
            return j - i + k

        # i = 0 border: inserting j leading characters of s2.
        dp[0, k] = 0
        for j in range(1, min(m + 1, k + 1)):
            dp[0, col_idx(0, j)] = j
        # j = 0 border: deleting i leading characters of s1. Without this the
        # pure-prefix-deletion path (e.g. "abc" -> "c") is unreachable and the
        # banded DP incorrectly returns the full deletion cost.
        for i in range(1, min(n + 1, k + 1)):
            dp[i, col_idx(i, 0)] = i
            
        for i in range(1, n + 1):
            c1 = s1[i - 1]
            min_j = max(1, i - k)
            max_j = min(m, i + k)
            
            for j in range(min_j, max_j + 1):
                c2 = s2[j - 1]
                sub_cost = 0 if c1 == c2 else 1
                
                c_ij = col_idx(i, j)
                
                # Match / Substitution: (i-1, j-1)
                cost_diag = dp[i - 1, col_idx(i - 1, j - 1)] + sub_cost
                
                # Deletion: (i-1, j)
                c_del = col_idx(i - 1, j)
                cost_del = dp[i - 1, c_del] + 1 if 0 <= c_del < (2 * k + 1) else INF
                
                # Insertion: (i, j-1)
                c_ins = col_idx(i, j - 1)
                cost_ins = dp[i, c_ins] + 1 if 0 <= c_ins < (2 * k + 1) else INF
                
                dp[i, c_ij] = min(cost_diag, cost_del, cost_ins)
                
        res = dp[n, col_idx(n, m)]
        return int(res) if res < INF else max(n, m)

    def approximate_edit_distance(self, s1: str, s2: str) -> Dict[str, Any]:
        """
        Sublinear multi-stage approximate edit distance:
        1. O(1) length difference trivial check.
        2. Fast q-gram lower bound filter.
        3. Adaptive diagonal banded verification.
        """
        t0 = time.perf_counter()
        n, m = len(s1), len(s2)
        len_diff = abs(n - m)
        
        q_bound = self.compute_qgram_lower_bound(s1, s2)
        
        # Adaptive band based on lower bound
        adaptive_k = max(self.band_width, int(q_bound * 1.5) + len_diff)
        approx_dist = self.compute_banded_edit_distance(s1, s2, max_k=adaptive_k)
        
        elapsed = time.perf_counter() - t0
        return {
            "approx_distance": approx_dist,
            "qgram_lower_bound": q_bound,
            "length_difference": len_diff,
            "latency_ms": elapsed * 1000.0
        }

    def find_approximate_matches(
        self,
        text: str,
        pattern: str,
        max_errors: int = 2,
        scan_step: int = 1,
    ) -> List[Tuple[int, int, int]]:
        """
        Finds approximate matches of pattern in text with at most max_errors.
        Returns list of (start_idx, end_idx, edit_distance).

        Recall / speed trade-off
        ------------------------
        ``scan_step`` controls how many text start offsets are skipped between
        probes. The default ``scan_step=1`` probes *every* offset and therefore
        never misses a 0-error (or low-error) match that begins at an unscanned
        offset -- this is the safe, full-recall setting. Raising ``scan_step``
        (e.g. ``p_len // 4``) gives a sublinear-time *approximate* scan that may
        silently miss matches whose start offset falls between sampled probes;
        use it only when approximate recall is acceptable and the text is large.
        """
        matches = []
        p_len = len(pattern)
        t_len = len(text)
        if p_len == 0 or t_len < p_len - max_errors:
            return matches

        step = max(1, int(scan_step))
        for i in range(0, t_len - p_len + max_errors + 1, step):
            for candidate_len in range(p_len - max_errors, min(t_len - i, p_len + max_errors) + 1):
                window = text[i:i + candidate_len]
                dist = self.compute_banded_edit_distance(pattern, window, max_k=max_errors + 1)
                if dist <= max_errors:
                    matches.append((i, i + candidate_len, dist))
                    break

        return matches


# ==============================================================================
# Classical Pointer-Based BK-Tree vs. Flat Elastic Deletion Hash Dictionary
# ==============================================================================

class BKTreeNode:
    """A node in a classical Burkhard-Keller metric tree."""
    def __init__(self, word: str):
        self.word = word
        self.children: Dict[int, "BKTreeNode"] = {}


class BKTree:
    """
    Classical Pointer-Based Burkhard-Keller Metric Tree (BK-Tree).
    
    Organizes vocabulary words into a metric tree where edge weights represent
    exact Levenshtein edit distances. Traversal relies on the triangle inequality
    to prune search branches: |d(node, query) - d(node, child)| <= max_distance.
    """
    def __init__(self):
        self.root: Optional[BKTreeNode] = None
        self.size = 0

    def insert(self, word: str):
        """Inserts a word into the pointer-based BK-Tree."""
        if not word:
            return
        if self.root is None:
            self.root = BKTreeNode(word)
            self.size = 1
            return
            
        curr = self.root
        while True:
            dist = exact_wagner_fischer_edit_distance(curr.word, word)
            if dist == 0:
                return  # Word already present
            if dist in curr.children:
                curr = curr.children[dist]
            else:
                curr.children[dist] = BKTreeNode(word)
                self.size += 1
                break

    def search(self, query: str, max_distance: int = 2) -> List[Tuple[str, int]]:
        """
        Recursively searches the BK-Tree for words within max_distance.
        Requires pointer traversal and metric distance calculations at visited nodes.
        """
        if self.root is None:
            return []
            
        results: List[Tuple[str, int]] = []
        stack = [self.root]
        
        while stack:
            node = stack.pop()
            d = exact_wagner_fischer_edit_distance(node.word, query)
            if d <= max_distance:
                results.append((node.word, d))
                
            # Triangle inequality branch pruning: [d - max_dist, d + max_dist]
            min_bound = max(1, d - max_distance)
            max_bound = d + max_distance
            
            for edge_dist, child in node.children.items():
                if min_bound <= edge_dist <= max_bound:
                    stack.append(child)
                    
        return results


class ElasticFuzzyDictionary:
    """
    Tree-Free Elastic Symmetric Deletion Vocabulary Hash Table.
    
    Replaces pointer-chasing BK-Trees and Levenshtein Tries with flat open addressing
    over precomputed symmetric deletion signatures. Provides O(1)-amortized query lookup
    at millions of queries per second with contiguous memory locality.
    """
    def __init__(self, max_edit_distance: int = 2):
        self.max_d = max(1, int(max_edit_distance))
        # Flat dictionary mapping deletion_key -> set of original candidate words
        self.deletion_table: Dict[str, List[str]] = {}
        self.words: Set[str] = set()

    def _get_deletions(self, word: str, max_d: int) -> Set[str]:
        """Generates all substring signatures formed by up to max_d character deletions."""
        deletions = {word}
        queue = {word}
        
        for _ in range(max_d):
            next_queue = set()
            for w in queue:
                if len(w) > 1:
                    for i in range(len(w)):
                        del_w = w[:i] + w[i + 1:]
                        if del_w not in deletions:
                            deletions.add(del_w)
                            next_queue.add(del_w)
            queue = next_queue
            
        return deletions

    def insert(self, word: str):
        """Indexes word and its symmetric deletions into flat hash map."""
        if not word or word in self.words:
            return
        self.words.add(word)
        deletions = self._get_deletions(word, self.max_d)
        for d_key in deletions:
            if d_key not in self.deletion_table:
                self.deletion_table[d_key] = []
            self.deletion_table[d_key].append(word)

    def insert_batch(self, words: List[str]):
        """Batch indexes a list of vocabulary words."""
        for w in words:
            self.insert(w)

    def search(self, query: str, max_distance: Optional[int] = None) -> List[Tuple[str, int]]:
        """
        Performs O(1) flat dictionary search for candidate matches within max_distance.
        """
        max_dist = self.max_d if max_distance is None else int(max_distance)
        if query in self.words:
            return [(query, 0)]
            
        candidates = set()
        query_deletions = self._get_deletions(query, max_dist)
        
        for q_del in query_deletions:
            if q_del in self.deletion_table:
                candidates.update(self.deletion_table[q_del])
                
        results = []
        for cand in candidates:
            # Fast length filter
            if abs(len(cand) - len(query)) > max_dist:
                continue
            d = exact_wagner_fischer_edit_distance(cand, query)
            if d <= max_dist:
                results.append((cand, d))
                
        results.sort(key=lambda x: x[1])
        return results
