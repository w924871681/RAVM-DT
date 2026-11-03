"""D2c gate-side capability consumer. Imports NO Service/label/authority provider.
In actual integration, bind ORIGINAL, UNMODIFIED authenticated CMC per-block gate callable.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
from d2c_core import VerifiedFeedbackView, view_message
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from cryptography.exceptions import InvalidSignature


@dataclass(frozen=True, slots=True)
class GateOutcome:
    action: bool
    fallback: bool
    reason: str
    original_called: bool


def d2c_gate(*, view: VerifiedFeedbackView, original_gate: Callable[[],bool],
             trusted_verifier_public_key: Ed25519PublicKey, expected_scope: tuple[str,str,str],
             expected_stage: int,
             research_candidate: Callable[[tuple],bool]|None = None) -> GateOutcome:
    """Fail closed TO EXACT ORIGINAL FUNCTION. This is an interface acceptance test,
    not an implementation or authorization of any D2b candidate algorithm.
    """
    # Fail-closed for any forged / replayed verifier output, even if attested=True.
    try:
        trusted_verifier_public_key.verify(bytes.fromhex(view.attestation_signature), view_message(view))
        valid_view=((view.run,view.center,view.stream)==expected_scope and
                    view.current_stage==expected_stage and
                    view.gate_start_seq>0)
    except (ValueError,InvalidSignature,AttributeError,TypeError):
        valid_view=False
    original_action=bool(original_gate())
    if not valid_view:
        return GateOutcome(original_action,True,"UNVERIFIED_VIEW_FALLBACK",True)
    if not view.attested or not view.audits:
        return GateOutcome(original_action,True,view.reason,True)
    # Explicitly NO research formula applied by default.
    if research_candidate is None:
        return GateOutcome(original_action,True,'NO_RESEARCH_RULE_ENABLED',True)
    # Only prevalidated immutable scalar audits are available to this callable.
    try:
        research_action=bool(research_candidate(view.audits))
    except Exception:
        return GateOutcome(original_action,True,'RESEARCH_CALLBACK_FAILURE_FALLBACK',True)
    return GateOutcome(research_action,False,'ISOLATED_RESEARCH_BRANCH_ONLY',True)
