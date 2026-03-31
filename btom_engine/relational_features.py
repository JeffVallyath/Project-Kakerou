"""Relational Feature Extractor — second-order ToM features from the social graph.

Computes 16 features per player from the FactRegistry, capturing cross-player
dynamics: who accuses whom, who defends known mafia, whose claims conflict.

These features use ground truth labels for some computations (accusation_accuracy,
defense_of_mafia_rate). During cross-validation, only training fold labels may be
used — test fold labels must be treated as unknown.
"""

from __future__ import annotations

import numpy as np
from collections import Counter
from typing import Any

from btom_engine.fact_registry import FactRegistry


RELATIONAL_FEATURE_NAMES = [
    # Accusation (4)
    "rel_accusation_rate",
    "rel_accused_by_rate",
    "rel_accusation_accuracy",
    "rel_accusation_scatter",
    # Defense (3)
    "rel_defense_of_mafia_rate",
    "rel_self_defense_rate",
    "rel_defended_by_town_rate",
    # Vote (3)
    "rel_vote_with_mafia_rate",
    "rel_vote_flip_rate",
    "rel_bandwagon_rate",
    # Claim conflict (3)
    "rel_conflicting_claims",
    "rel_unique_role_conflict",
    "rel_claim_verified_rate",
    # Social graph (3)
    "rel_centrality",
    "rel_cluster_with_mafia",
    "rel_isolation_score",
]


def extract_relational_features(
    player_id: str,
    registry: FactRegistry,
    labels: dict[str, bool],
    n_turns: int,
) -> dict[str, float]:
    """Extract 16 relational features for one player.

    Args:
        player_id: the player to compute features for
        registry: shared FactRegistry with all players' facts
        labels: dict[player_id -> is_mafia] (ground truth, training fold only)
        n_turns: total turns this player participated in

    Returns:
        dict of feature_name -> value. Uses np.nan for features that
        can't be computed (insufficient data).
    """
    n_turns = max(n_turns, 1)
    feat: dict[str, float] = {}

    pid = player_id.lower()

    # All known players and their labels
    known_mafia = {p.lower() for p, is_m in labels.items() if is_m}
    known_town = {p.lower() for p, is_m in labels.items() if not is_m}

    # === ACCUSATION FEATURES (4) ===

    my_accusations = registry.get_accusations_by(player_id)
    accusations_of_me = registry.get_accusations_of(player_id)

    # accusation_rate: how often do I accuse others (per turn)
    feat["rel_accusation_rate"] = len(my_accusations) / n_turns

    # accused_by_rate: how often am I accused (per turn)
    feat["rel_accused_by_rate"] = len(accusations_of_me) / n_turns

    # accusation_accuracy: of people I accused, what % were actually mafia
    accused_targets = [a.target.lower() for a in my_accusations]
    accused_with_known_label = [t for t in accused_targets if t in known_mafia or t in known_town]
    if accused_with_known_label:
        correct = sum(1 for t in accused_with_known_label if t in known_mafia)
        feat["rel_accusation_accuracy"] = correct / len(accused_with_known_label)
    else:
        feat["rel_accusation_accuracy"] = np.nan

    # accusation_scatter: how many DIFFERENT players do I accuse
    unique_targets = set(accused_targets)
    feat["rel_accusation_scatter"] = len(unique_targets) / max(len(my_accusations), 1)

    # === DEFENSE FEATURES (3) ===

    my_defenses = registry.get_defenses_by(player_id)
    defenses_of_me = registry.get_defenses_of(player_id)

    # defense_of_mafia_rate: of players I defended, what % turned out to be mafia
    defended_targets = [d.defended.lower() for d in my_defenses if d.defended.lower() != pid]
    defended_with_known = [t for t in defended_targets if t in known_mafia or t in known_town]
    if defended_with_known:
        mafia_defended = sum(1 for t in defended_with_known if t in known_mafia)
        feat["rel_defense_of_mafia_rate"] = mafia_defended / len(defended_with_known)
    else:
        feat["rel_defense_of_mafia_rate"] = np.nan

    # self_defense_rate: how often do I defend myself
    self_defenses = [d for d in my_defenses if d.defended.lower() == pid]
    feat["rel_self_defense_rate"] = len(self_defenses) / n_turns

    # defended_by_town_rate: of players defending me, what % are town
    defenders = [d.defender.lower() for d in defenses_of_me if d.defender.lower() != pid]
    defenders_with_known = [d for d in defenders if d in known_mafia or d in known_town]
    if defenders_with_known:
        town_defenders = sum(1 for d in defenders_with_known if d in known_town)
        feat["rel_defended_by_town_rate"] = town_defenders / len(defenders_with_known)
    else:
        feat["rel_defended_by_town_rate"] = np.nan

    # === VOTE FEATURES (3) ===

    my_votes = registry.get_votes_by(player_id)

    # vote_with_mafia_rate: do my vote targets align with what mafia voted for
    if my_votes and known_mafia:
        mafia_vote_targets = Counter()
        for v in registry.votes:
            if v.voter.lower() in known_mafia:
                mafia_vote_targets[v.target.lower()] += 1
        if mafia_vote_targets:
            aligned = sum(1 for v in my_votes if v.target.lower() in mafia_vote_targets)
            feat["rel_vote_with_mafia_rate"] = aligned / len(my_votes)
        else:
            feat["rel_vote_with_mafia_rate"] = np.nan
    else:
        feat["rel_vote_with_mafia_rate"] = np.nan

    # vote_flip_rate: how often do I change vote targets
    if len(my_votes) >= 2:
        flips = sum(1 for i in range(1, len(my_votes))
                    if my_votes[i].target.lower() != my_votes[i-1].target.lower())
        feat["rel_vote_flip_rate"] = flips / (len(my_votes) - 1)
    else:
        feat["rel_vote_flip_rate"] = np.nan

    # bandwagon_rate: do I vote for already-popular targets
    if my_votes:
        bandwagons = 0
        for v in my_votes:
            # Count how many votes this target already had before my vote
            prior_votes = sum(1 for v2 in registry.votes
                              if v2.target.lower() == v.target.lower()
                              and v2.turn < v.turn
                              and v2.voter.lower() != pid)
            if prior_votes >= 2:  # 2+ prior votes = bandwagon
                bandwagons += 1
        feat["rel_bandwagon_rate"] = bandwagons / len(my_votes)
    else:
        feat["rel_bandwagon_rate"] = np.nan

    # === CLAIM CONFLICT FEATURES (3) ===

    my_role_claims = registry.get_role_claims(player_id)

    # conflicting_claims: how many of my role claims conflict with others
    conflicts = 0
    for claim in my_role_claims:
        others_with_same = registry.get_all_players_claiming_role(claim.claimed_role)
        others_with_same = [p for p in others_with_same if p.lower() != pid]
        if others_with_same:
            conflicts += 1
    feat["rel_conflicting_claims"] = float(conflicts)

    # unique_role_conflict: did I claim a unique role that someone else also claimed
    unique_roles = {"seer", "cop", "detective", "investigator", "doctor", "medic",
                    "guard", "witch", "hunter", "vigilante", "oracle", "tracker", "watcher"}
    role_conflict = 0
    for claim in my_role_claims:
        if claim.claimed_role.lower() in unique_roles:
            others = registry.get_all_players_claiming_role(claim.claimed_role)
            if len(others) > 1:  # multiple claimants for unique role
                role_conflict = 1
                break
    feat["rel_unique_role_conflict"] = float(role_conflict)

    # claim_verified_rate: placeholder — hard to compute without game event log
    feat["rel_claim_verified_rate"] = np.nan

    # === SOCIAL GRAPH FEATURES (3) ===

    # centrality: how connected am I (accusations made + received + defenses)
    all_interactions = (len(my_accusations) + len(accusations_of_me) +
                        len(my_defenses) + len(defenses_of_me) + len(my_votes))
    total_interactions = max(len(registry.accusations) + len(registry.defenses) + len(registry.votes), 1)
    feat["rel_centrality"] = all_interactions / total_interactions

    # cluster_with_mafia: do I interact more with mafia members than expected
    if known_mafia and known_town:
        mafia_interactions = 0
        total_player_interactions = 0
        for a in my_accusations:
            if a.target.lower() in known_mafia:
                mafia_interactions += 1
            if a.target.lower() in (known_mafia | known_town):
                total_player_interactions += 1
        for d in my_defenses:
            if d.defended.lower() in known_mafia and d.defended.lower() != pid:
                mafia_interactions += 1
            if d.defended.lower() in (known_mafia | known_town) and d.defended.lower() != pid:
                total_player_interactions += 1
        if total_player_interactions > 0:
            expected_mafia_rate = len(known_mafia) / (len(known_mafia) + len(known_town))
            actual_mafia_rate = mafia_interactions / total_player_interactions
            feat["rel_cluster_with_mafia"] = actual_mafia_rate - expected_mafia_rate
        else:
            feat["rel_cluster_with_mafia"] = np.nan
    else:
        feat["rel_cluster_with_mafia"] = np.nan

    # isolation_score: am I unusually disconnected
    unique_interaction_partners = set()
    for a in my_accusations:
        unique_interaction_partners.add(a.target.lower())
    for d in my_defenses:
        if d.defended.lower() != pid:
            unique_interaction_partners.add(d.defended.lower())
    for a in accusations_of_me:
        unique_interaction_partners.add(a.accuser.lower())

    n_players = max(len(registry.player_names), 1)
    feat["rel_isolation_score"] = 1.0 - (len(unique_interaction_partners) / max(n_players - 1, 1))

    return feat
