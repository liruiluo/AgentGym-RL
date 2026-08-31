from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Literal
from transformers import PreTrainedTokenizer
import torch

# Rollout thinking is off by default so legacy runs keep emitting bare actions.
# Set AGENTMEMORY_ENABLE_THINKING=1 to let the model reason in a <think> block
# before acting -- this task needs it, since choosing the right product means
# reading candidate prices/ratings and comparing them against the Goal, which a
# bare action cannot express.
def _agentmemory_thinking_enabled() -> bool:
    return os.environ.get("AGENTMEMORY_ENABLE_THINKING", "0").strip().lower() in ("1", "true", "yes", "on")


def _agentmemory_reasoning_enabled() -> bool:
    """Allow explicit ReAct reasoning without enabling Qwen native thinking."""
    return os.environ.get("AGENTMEMORY_ALLOW_REASONING", "0").strip().lower() in ("1", "true", "yes", "on")


def agentmemory_ltm_inventory_mode() -> str:
    mode = os.environ.get("AGENTMEMORY_LTM_INVENTORY_MODE", "hidden").strip()
    if mode not in ("hidden", "keys"):
        raise ValueError(
            "AGENTMEMORY_LTM_INVENTORY_MODE must be 'hidden' or 'keys'."
        )
    return mode


AGENTMEMORY_MEMORY_PROMPT_MODES = (
    "legacy",
    "neutral",
    "neutral_horizon",
    "neutral_horizon_responsibility",
    "latent_preference_sop",
    "selective_memory_sop",
    "natural_filesystem",
)

NATURAL_FILESYSTEM_PROMPT_MODE = "natural_filesystem"
NATURAL_FILESYSTEM_SURFACE = (
    "agentmemory_webshop_procedural_natural_chain_filesystem_v2"
)
RECENCY_OVERRIDE_FILESYSTEM_SURFACE = (
    "agentmemory_webshop_recency_override_filesystem_v2"
)
LATENT_PREFERENCE_FILESYSTEM_SURFACE = (
    "agentmemory_webshop_latent_preference_filesystem_v2"
)
DISTRACTOR_ROBUSTNESS_FILESYSTEM_SURFACE = (
    "agentmemory_webshop_distractor_robustness_filesystem_v2"
)
COMPOSITIONAL_RECALL_FILESYSTEM_SURFACE = (
    "agentmemory_webshop_compositional_recall_filesystem_v2"
)
NEGATIVE_CONSTRAINT_FILESYSTEM_SURFACE = (
    "agentmemory_webshop_negative_constraint_filesystem_v2"
)
INTENT_CLARIFICATION_FILESYSTEM_SURFACE = (
    "agentmemory_webshop_intent_clarification_filesystem_v2"
)
SELECTIVE_MEMORY_USE_FILESYSTEM_SURFACE = (
    "agentmemory_webshop_selective_memory_use_filesystem_v2"
)
FILESYSTEM_SURFACES = frozenset(
    {
        NATURAL_FILESYSTEM_SURFACE,
        RECENCY_OVERRIDE_FILESYSTEM_SURFACE,
        LATENT_PREFERENCE_FILESYSTEM_SURFACE,
        DISTRACTOR_ROBUSTNESS_FILESYSTEM_SURFACE,
        COMPOSITIONAL_RECALL_FILESYSTEM_SURFACE,
        NEGATIVE_CONSTRAINT_FILESYSTEM_SURFACE,
        INTENT_CLARIFICATION_FILESYSTEM_SURFACE,
        SELECTIVE_MEMORY_USE_FILESYSTEM_SURFACE,
    }
)


def agentmemory_memory_prompt_mode() -> str:
    mode = os.environ.get("AGENTMEMORY_MEMORY_PROMPT_MODE", "legacy").strip()
    if mode not in AGENTMEMORY_MEMORY_PROMPT_MODES:
        raise ValueError(
            "AGENTMEMORY_MEMORY_PROMPT_MODE must be one of: "
            + ", ".join(AGENTMEMORY_MEMORY_PROMPT_MODES)
            + "."
        )
    return mode


AGENTMEMORY_ACTION_LISTING_MODES = ("separate", "unified")


def agentmemory_action_listing_mode() -> str:
    mode = os.environ.get("AGENTMEMORY_ACTION_LISTING_MODE", "separate").strip()
    if mode not in AGENTMEMORY_ACTION_LISTING_MODES:
        raise ValueError(
            "AGENTMEMORY_ACTION_LISTING_MODE must be 'separate' or 'unified'."
        )
    return mode


# The system prompt is built from three parts. The intro and the action-space
# contract are identical in both modes; only the reply-format rule differs, so
# that the rule never contradicts whether the chat template opened a <think>
# block (forbidding <think> while the template opens one would be self-defeating).
_AGENTMEMORY_INTRO = (
    "You are acting inside AgentMemoryGym, a native WebShop bundled-shopping environment. "
)

# No-thinking mode: the entire reply must be one bare action.
_AGENTMEMORY_REPLY_RULE_NO_THINKING = (
    "Reply with exactly one executable action and nothing else: either one native browser "
    "action or one uppercase memory-tool JSON action. Output excludes angle-bracket "
    "placeholders, markdown, explanations, Thought/Action labels, and <think> blocks. "
)

# Thinking mode: optional reasoning inside one <think> block, then one bare action.
_AGENTMEMORY_REPLY_RULE_THINKING = (
    "You may first reason inside a single <think>...</think> block. After the closing "
    "</think>, reply with exactly one executable action and nothing else: either one native "
    "browser action or one uppercase memory-tool JSON action. Apart from that optional "
    "<think> block, output excludes angle-bracket placeholders, markdown, explanations, and "
    "Thought/Action labels. "
)

# ReAct reasoning mode: the chat template keeps native thinking disabled, while
# the policy emits a short Thought plus the one Action executed by the env.
_AGENTMEMORY_REPLY_RULE_REASONING = (
    "Reply with exactly two labeled fields. Write `Thought:` followed by brief free-form "
    "reasoning, then write `Action:` followed by exactly one executable action: either one "
    "native browser action or one uppercase memory-tool JSON action. The environment executes "
    "only the action after the final `Action:` label, while PPO trains the complete sampled "
    "Thought-and-Action response. Output excludes markdown and <think> blocks. "
)

_AGENTMEMORY_ACTION_CONTRACT = (
    "Native browser actions use square-bracket syntax. search[keywords] runs a catalog "
    "search whose keywords are concrete product wording such as a visible product name or "
    "title; a bare category word or attribute alone matches little. click[value] clicks one "
    "currently displayed clickable value, exactly as shown in the available-actions list: an "
    "asin opens that product page, and the page also exposes navigation such as "
    "click[Back to Search], click[< Prev], click[Next >], click[Description], click[Features], "
    "click[Reviews], option values, and click[Buy Now]. A product page shows title, price, "
    "rating, sub-pages, and selectable options. click[Buy Now] on the open product commits the "
    "purchase of the current shopping session; a correct purchase advances to the next session "
    "and an incorrect purchase ends the episode with reward -0.01 and no retry. The visible "
    "available-actions list enumerates the clickable values valid on the current page. "
    "Memory tools use one uppercase name followed by one JSON object. ADD requires key:string "
    "and value:string and returns a new memory_id while storing exactly the text you wrote. "
    "UPDATE requires memory_id:string and value:string and replaces that memory value. DELETE "
    "requires memory_id:string and removes it. RETRIEVE accepts exactly one lookup field: "
    "query:string for BM25 text matching with optional top_k:int (default 3), or "
    "memory_id:string for exact readback of that entry. It reads only text you previously wrote "
    "to long-term memory with ADD (facts carried over from earlier sessions), not the current "
    "page or catalog, exposing retrieved entries as visible C# items. "
    "SUMMARY requires text:string and a non-empty source_ids:list[string] of visible S#/C# ids "
    "and replaces active context with that summary. FILTER requires exactly one non-empty "
    "keep_ids:list[string] or drop_ids:list[string], plus scope set to active, session, or all, "
    "and only changes visible S#/C# context. Current-session browser trace is shown as S# "
    "items and retrieved or summarized memory as C# items. Current-session trace clears when a "
    "purchase advances the session. Long-term memory persists across shopping sessions and "
    "remains hidden until RETRIEVE exposes it."
)

_AGENTMEMORY_MEMORY_LIFECYCLE = (
    "A successful purchase clears the current session's page and short-term trace. Once "
    "you have selected the product for the current session, use ADD before click[Buy Now] "
    "to save one concise memory containing that product's identity and any visible "
    "attributes needed for later compatibility decisions. At the start of every later "
    "shopping session, use RETRIEVE to expose the relevant prior-purchase memories before "
    "choosing a compatible product. The environment does not perform these memory actions "
    "for you, and it does not reject an otherwise correct purchase when ADD was skipped."
)

_AGENTMEMORY_TASK_HORIZON = (
    "This episode has six sequential shopping sessions. Later-session compatibility "
    "constraints may refer to products purchased in earlier sessions."
)

_AGENTMEMORY_CROSS_SESSION_MEMORY_RESPONSIBILITY = (
    "Across shopping sessions, you are responsible for preserving and accessing any "
    "facts needed for later decisions."
)

_AGENTMEMORY_LATENT_PREFERENCE_SOP = (
    "Each early evidence session may show which approved listing a customer "
    "confirmed. Treat each confirmed choice as preference evidence. Preserve the "
    "confirmed listing and its visible distinguishing attributes. After a confirmed "
    "choice is visible, use ADD before click[Buy Now] to store that evidence or "
    "create a customer-profile memory containing the customer, preference axis, and "
    "inferred value. When a customer-profile memory already exists, first retrieve "
    "its exact memory_id and use UPDATE to incorporate additional evidence without "
    "discarding prior support. Do not assume a fixed number of examples is always "
    "sufficient; infer a preference only when the visible confirmed choices support "
    "it. At the start of every later shopping session, use RETRIEVE to expose the "
    "relevant confirmed-choice evidence or customer profile. In later application "
    "sessions, apply the retrieved preference when choosing between approved "
    "listings. The environment does not perform these memory actions for you, and it "
    "does not reject an otherwise correct purchase when ADD was skipped."
)

_AGENTMEMORY_SELECTIVE_MEMORY_SOP = (
    "First decide whether the current request already states every attribute needed "
    "to choose between its approved listings. When the current request is complete, "
    "follow it directly: explicit current requirements override profile history, and "
    "you should not ADD or RETRIEVE merely by habit. When the current request omits "
    "the customer's profile preference, use RETRIEVE to expose the saved current "
    "profile before choosing. Store new memory only when the episode provides new "
    "information that a later session will actually need."
)

# The filesystem surface deliberately has its own prompt family.  Reusing the
# legacy action contract here would silently teach the policy that a dedicated
# memory API exists, which is exactly the scaffold this surface is meant to
# remove.
_AGENTMEMORY_FILESYSTEM_REPLY_RULE_NO_THINKING = (
    "Reply with exactly one executable action and nothing else. The first non-whitespace "
    "text must be exactly one native browser action, one canonical shell_command action, "
    "or one multiline apply_patch action. The canonical shell form is the literal prefix "
    "shell_command, one space, then one JSON object. A bare JSON object, markdown code "
    "fence, explanation, Thought/Action label, or <think> block is invalid. "
)
_AGENTMEMORY_FILESYSTEM_REPLY_RULE_THINKING = (
    "You may first reason inside a single <think>...</think> block. After the closing "
    "</think>, reply with exactly one executable action and nothing else. The action must "
    "be one native browser action, one canonical shell_command action, or one multiline "
    "apply_patch action. The canonical shell form is the literal prefix shell_command, one "
    "space, then one JSON object. A bare JSON object, markdown code fence, explanation, or "
    "Thought/Action label after </think> is invalid. "
)
_AGENTMEMORY_FILESYSTEM_REPLY_RULE_REASONING = (
    "Reply with exactly two labeled fields. Write `Thought:` followed by brief free-form "
    "reasoning, then write `Action:` followed by exactly one executable action: one native "
    "browser action, one canonical shell_command action, or one multiline apply_patch "
    "action. The canonical shell form is the literal prefix shell_command, one space, then "
    "one JSON object. The environment executes only the complete action after the final "
    "`Action:` label, while PPO trains the complete sampled Thought-and-Action response. "
    "Do not put the action in a markdown code fence or emit a bare JSON object. "
)
_AGENTMEMORY_NO_WORKSPACE_REPLY_RULE_NO_THINKING = (
    "Reply with exactly one executable native browser action and nothing else. Output "
    "excludes markdown, explanations, Thought/Action labels, and <think> blocks. "
)
_AGENTMEMORY_NO_WORKSPACE_REPLY_RULE_THINKING = (
    "You may first reason inside a single <think>...</think> block. After the closing "
    "</think>, reply with exactly one executable native browser action and nothing else. "
    "Apart from that optional <think> block, output excludes markdown, explanations, and "
    "Thought/Action labels. "
)
_AGENTMEMORY_NO_WORKSPACE_REPLY_RULE_REASONING = (
    "Reply with exactly two labeled fields. Write `Thought:` followed by brief free-form "
    "reasoning, then write `Action:` followed by exactly one executable native browser "
    "action. The environment executes only the complete action after the final `Action:` "
    "label, while PPO trains the complete sampled Thought-and-Action response. Output "
    "excludes markdown and <think> blocks. "
)
_AGENTMEMORY_FILESYSTEM_ACTION_CONTRACT = (
    "Native browser actions use square-bracket syntax. search[keywords] searches the visible "
    "catalog. click[value] clicks one value in the current available-actions list; an ASIN "
    "opens that product page, and click[Buy Now] commits the current shopping session only "
    "when Buy Now is currently shown in the available-actions list. From the Search page, "
    "the required sequence is search[the approved listing's complete visible title], then "
    "click[its displayed ASIN], then click[Buy Now] after the exact product page opens. A "
    "correct purchase advances to the next session; an incorrect purchase terminates the "
    "episode. A product page exposes its title, price, rating, sub-pages, options, and the "
    "Buy Now action. Two Codex-style workspace tools operate on a private episode workspace. "
    'The canonical shell form is exactly shell_command {"command":"rg -n pattern .",'
    '"workdir":".","timeout_ms":10000}; the literal prefix and separating space are '
    "required, and a bare JSON object is invalid. It runs one shell command; command is "
    "required and workdir and timeout_ms are optional. "
    "The shell includes ordinary file utilities and pinned rg, has no network, and runs with "
    "bounded time, output, processes, storage, and privileges. apply_patch is followed on the "
    "next line by one Codex patch beginning with *** Begin Patch and ending with *** End Patch; "
    "it supports Add File, Update File, Delete File, and Move to. "
    "The workspace persists across shopping sessions within this episode, permits ordinary "
    "nested files, and is reset between episodes. Workspace actions have zero task reward "
    "and are optional; use them when a later decision needs a fact that is no longer in the "
    "current observation. There is no host-path access and no dedicated memory API. Use one "
    "browser or workspace action per turn."
)
_AGENTMEMORY_FILESYSTEM_MEMORY_GUIDANCE = (
    " The workspace starts empty and contains only files that you create; it is not a "
    "catalog, hidden cache, or automatic session log. In the first shopping session there "
    "is no preceding purchase or note: do not run a workspace read, and start with "
    "search[...] for the exact approved listing selected by the explicit starting request. "
    "The read-and-retry rule below applies only after a purchase has advanced the episode "
    "to a later session whose current prompt says its choice depends on the preceding "
    "purchase. After click[Buy Now] advances sessions 1 through 5, the environment issues "
    "a checkpoint request before replacing the context. Answer that request with exactly one "
    "ordinary executable workspace action to overwrite `.agent_memory/CONTINUATION.md`. "
    "Preserve the exact `Confirmed <field>: <value>` evidence for the completed purchase, "
    "completed purchases, remaining budget, and the next browser action. Do not create a "
    "duplicate handoff note before buying. Other workspace documents remain optional and do "
    "not replace the canonical checkpoint. Shopping-session transitions use "
    "latest-observation-only input, so the earlier action reply and checkpoint contents are "
    "not shown after advancing. In a session whose choice depends on the preceding purchase, "
    "the first action must use shell_command to read the canonical continuation checkpoint "
    "exactly with `cat .agent_memory/CONTINUATION.md`; do not search, click, or write a "
    "replacement fact until the preceding session's saved evidence has appeared in shell "
    "output. If that exact read fails or returns empty output, retry the exact `cat` command "
    "once. After one successful non-empty checkpoint read, continue with the required browser "
    "action and do not run `cat` or `rg` again in that session; never loop on an unchanged "
    "successful read. An empty result is never "
    "permission to guess, search, click, or write a replacement fact. Never infer or recreate "
    "the missing value from the choice table. Listing a directory, scanning the workspace, or "
    "reading a different file is not reading the checkpoint. After the preceding session's "
    "saved evidence is printed, follow this order without skipping steps: choose the branch "
    "only from that exact saved field and value; copy the chosen approved card's complete "
    "Product title into search[...] without shortening it; open only a result whose complete "
    "visible title exactly equals the card, including size, count, and pack qualifiers; then "
    "use click[Buy Now]. The environment supplies tool feedback on the "
    "next turn, so never append Result or feedback text to the action. Here is one generic "
    "Codex example unrelated to "
    "the shopping task; the two complete replies are separate turns and must never be emitted "
    "together. Earlier turn (complete reply):\n"
    "apply_patch\n"
    "*** Begin Patch\n"
    "*** Add File: .agent_memory/example.md\n"
    "+service port: 4317\n"
    "*** End Patch\n"
    "After the environment returns feedback, a later turn may locate and read that note when "
    "the fact and chosen filename are no longer visible. Later turn (complete reply):\n"
    "shell_command {\"command\":\"rg --hidden -n '^service port:' .\",\"workdir\":\".\","
    "\"timeout_ms\":10000}\n"
    "Choose filenames and note contents from the actual future need; the example value and "
    "path are illustrative, not requirements. Keep each reply to exactly one action."
)
_AGENTMEMORY_RECENCY_FILESYSTEM_MEMORY_GUIDANCE = (
    " The workspace starts empty and contains only files that you create; it is not a "
    "catalog, hidden cache, or automatic session log. Maintain one ordinary file as the "
    "current confirmed user-preference record. Evidence and update questions provide one "
    "exact policy-visible record line beginning with `Current preference:`. Before advancing "
    "beyond such a session, copy that exact line verbatim into the current-state file. "
    "Do not rename its field, infer a value from product titles, or substitute other state. "
    "The user may explicitly replace that preference in a later session. When that happens, "
    "use apply_patch Update File on the existing current-state file so the new value replaces "
    "the old value; do not leave conflicting current and stale values in separate notes. If "
    "the path is no longer visible, discover it with `rg --hidden -n '^Current preference:' .` "
    "before editing. In every later application session, first use shell_command to print the "
    "current preference record, and choose only from the exact current value printed there. "
    "Do not infer the missing value from the choice table or reuse an older value. A merchandise "
    "category, product title, or option-table entry is not the preference record. If shell stdout "
    "is empty or the command fails, correct and retry the workspace read before any search or "
    "click; never guess the value. The environment supplies tool feedback on the next turn, so "
    "never append Result or feedback text to the action. To create the record, use exactly five "
    "physical lines: line 1 is exactly `apply_patch`; line 2 is exactly `*** Begin Patch`; line "
    "3 begins with `*** Add File: `, including the colon and following space, then a new relative "
    "path; line 4 begins with `+` immediately followed by the task's exact record; line 5 is "
    "exactly `*** End Patch`. Never combine two patch lines on one physical line. To replace the "
    "record, use exactly seven physical lines: line 1 is exactly `apply_patch`; line 2 is exactly "
    "`*** Begin Patch`; line 3 begins with `*** Update File: `, including the colon and following "
    "space, then the existing relative path; line 4 is exactly `@@`; line 5 begins with `-` "
    "immediately followed by the exact old record; line 6 begins with `+` immediately followed "
    "by the exact new task-provided record; line 7 is exactly `*** End Patch`. After any workspace "
    "syntax rejection or patch failure, assume the workspace is unchanged and fix and retry the "
    "workspace action before taking a browser action; only `Done!` proves that the patch succeeded. "
    "Run shell_command in a separate turn after tool feedback when the record must be read. No "
    "concrete preference value or filename is demonstrated here: choose both only from "
    "policy-visible task facts. Keep each reply to exactly one action."
)
_AGENTMEMORY_DISTRACTOR_FILESYSTEM_MEMORY_GUIDANCE = (
    " The workspace may start with harness-seeded ordinary profile notes. Those files are "
    "background records, not policy actions, hidden answers, or automatically trusted current "
    "state. In shopping session 0, the task provides one exact authoritative record line "
    "beginning with `Current preference:`. Before buying, copy that exact line verbatim into "
    "a new ordinary file using apply_patch; do not rewrite or delete the seeded notes. In every "
    "later session whose decision depends on the profile, first use shell_command to inspect "
    "the workspace with `rg --hidden -n -i '(current|preference|profile)' .`. Read the returned "
    "file contents and select the authoritative current record for the active customer and "
    "attribute axis. Ignore superseded history, another customer's record, and records about a "
    "different axis. Do not rely on filenames, directory order, or the number of files. Do not "
    "search or click until shell output contains the exact policy-authored `Current preference:` "
    "line. If it is absent, correct and retry the file read; never infer the missing value from "
    "the current choice table. Create the current record with exactly five physical lines: "
    "`apply_patch`, `*** Begin Patch`, one `*** Add File: ` line with a new relative path, one "
    "content line beginning with `+`, and `*** End Patch`. Run the write, later read, and browser "
    "actions on separate turns. Only `Done!` proves a patch succeeded, and workspace feedback "
    "must never be appended to the action. No concrete customer, axis, preference value, or "
    "filename is demonstrated here; take them only from policy-visible task facts. Keep each "
    "reply to exactly one action."
)
_AGENTMEMORY_LATENT_PREFERENCE_FILESYSTEM_MEMORY_GUIDANCE = (
    " The workspace starts empty and contains only files that you create; it is not a "
    "catalog, hidden answer, or automatic session log. Treat every confirmed choice as "
    "preference evidence, and preserve confirmed preference evidence in an ordinary workspace "
    "file before advancing when a later application session will need it. A later profile "
    "note may be a customer-profile memory: keep the customer identity, preference axis, "
    "and inferred value together, and retain the supporting confirmed choices instead of "
    "overwriting them with a bare label. Use shell_command to read the relevant note before "
    "searching or clicking, then apply the retrieved preference in later application sessions "
    "to the approved listings. If the note is absent or the field is not visible, read and "
    "retry rather than infer it from product titles or the current choice table. Workspace "
    "writes and reads are separate turns, have zero task reward, and are optional when the "
    "current request already contains everything needed. No concrete customer, preference "
    "axis, inferred value, or filename is demonstrated here; take them only from policy-visible "
    "task facts. Keep each reply to exactly one action."
)
_AGENTMEMORY_INTENT_CLARIFICATION_FILESYSTEM_MEMORY_GUIDANCE = (
    " The workspace starts empty and contains only files that you create; it is not a "
    "catalog, hidden answer, or automatic session log. The first shopping request is "
    "intentionally ambiguous: both approved listings satisfy the stated requirements, and "
    "the missing preference must be identified from the request and candidate attributes. "
    "ASK requires field:string. Use ASK {\"field\":\"...\"} as the generic clarification "
    "schema; infer and fill the missing field rather than copying a task-specific field from "
    "the action contract. ASK "
    "is available only in the first shopping session and is allowed once. The environment "
    "returns a CLARIFY observation; store the clarification in an ordinary workspace file "
    "before the first purchase when later sessions need it. In later sessions, use "
    "shell_command to read that note before choosing the matching approved listing. Do not "
    "purchase before a valid clarification, do not repeat ASK, and do not infer the answer "
    "from the choice table. Keep workspace writes and reads on separate turns and keep each "
    "reply to exactly one action."
)
_AGENTMEMORY_SELECTIVE_MEMORY_USE_FILESYSTEM_MEMORY_GUIDANCE = (
    " The workspace may start with one branch-conditioned ordinary profile file. It is "
    "harness-seeded background state, not a hidden answer and not a policy action. You must "
    "first decide whether the current request already states every attribute needed to choose between "
    "the approved listings. When it is complete, follow the current request directly and do "
    "not read the profile merely by habit; explicit current requirements override profile "
    "history, and do not write a redundant note. When the current request omits the preference, "
    "read the profile when the current request omits the preference using shell_command, copy "
    "the relevant policy-visible record into a new ordinary note if a later session needs it, "
    "and apply that preference to the approved listings. Do not rely on filenames or directory "
    "order, and do not infer a missing profile value from the choice table. Workspace actions "
    "have zero task reward and each reply contains exactly one action."
)
_AGENTMEMORY_COMPOSITIONAL_FILESYSTEM_MEMORY_GUIDANCE = (
    " The workspace starts empty and contains only files that you create; it is not a "
    "catalog, hidden cache, or automatic session log. This task exposes two separate "
    "relations. In shopping session 0, before buying, save the exact visible customer ID "
    "and active shopping profile token in one ordinary file as a line beginning with "
    "`Customer-to-profile:`. In shopping session 1, save both exact visible profile-token "
    "directory entries, including the attribute axis and values, in another ordinary file "
    "as a line beginning with `Profile-directory:`. Preserve the two hops separately; do "
    "not collapse them into a direct customer preference or infer either hop from product "
    "titles. In every session from session 2 onward, the first action must use shell_command "
    "to print both records with `rg --hidden -n '^(Customer-to-profile|Profile-directory):' .`. "
    "Do not search or click until shell output contains both records. Compose the exact chain "
    "customer -> active profile token -> attribute value, then choose the approved listing "
    "with that value. If either record is absent, correct and retry the file read; never infer "
    "the missing hop from the current choice table. Create each record with exactly five "
    "physical lines: `apply_patch`, `*** Begin Patch`, one `*** Add File: ` line with a new "
    "relative path, one content line beginning with `+`, and `*** End Patch`. Run file writes, "
    "reads, and browser actions on separate turns. Only `Done!` proves a patch succeeded, and "
    "workspace tool feedback must never be appended to the action. No concrete customer, "
    "profile token, attribute value, or filename is demonstrated here; take them only from "
    "policy-visible task facts. Keep each reply to exactly one action."
)
_AGENTMEMORY_NEGATIVE_FILESYSTEM_MEMORY_GUIDANCE = (
    " The workspace starts empty and contains only files that you create; it is not a "
    "catalog, hidden cache, or automatic session log. In shopping session 0, the customer "
    "states two standing never-accept values on one attribute axis. Before buying, save the "
    "exact axis and both exact forbidden values in one ordinary file as a line beginning with "
    "`Standing exclusions:`. Store the exclusions themselves, not only the currently allowed "
    "value or product title. In every later session, the first action must use shell_command "
    "to print the record with `rg --hidden -n '^Standing exclusions:' .`. Do not search or "
    "click until that exact record appears in shell output. Reject each candidate that matches "
    "either forbidden value and purchase the sole remaining approved listing. If the record is "
    "absent, correct and retry the file read; never infer the exclusions from the current choice "
    "table. Create the record with exactly five physical lines: `apply_patch`, `*** Begin Patch`, "
    "one `*** Add File: ` line with a new relative path, one content line beginning with `+`, "
    "and `*** End Patch`. Run the write, later read, and browser actions on separate turns. "
    "Only `Done!` proves the patch succeeded, and workspace tool feedback must never be appended "
    "to the action. No concrete axis, forbidden value, or filename is demonstrated here; take "
    "them only from policy-visible task facts. Keep each reply to exactly one action."
)
_AGENTMEMORY_NO_WORKSPACE_ACTION_CONTRACT = (
    "Native browser actions use square-bracket syntax. search[keywords] searches the visible "
    "catalog. click[value] clicks one value in the current available-actions list; an ASIN "
    "opens that product page, and click[Buy Now] commits the current shopping session. A "
    "correct purchase advances to the next session; an incorrect purchase terminates the "
    "episode. A product page exposes its title, price, rating, sub-pages, options, and the "
    "Buy Now action. This causal intervention provides no persistent workspace: "
    "shell_command and apply_patch are unavailable and invalid. Use one native browser "
    "action per turn."
)
AGENTMEMORY_ACTION_SYSTEM_PROMPT_NATURAL_FILESYSTEM = (
    "You are acting inside AgentMemoryGym, a native WebShop bundled-shopping environment "
    "with a persistent workspace. "
    + _AGENTMEMORY_FILESYSTEM_REPLY_RULE_NO_THINKING
    + _AGENTMEMORY_FILESYSTEM_ACTION_CONTRACT
    + _AGENTMEMORY_FILESYSTEM_MEMORY_GUIDANCE
)
AGENTMEMORY_ACTION_SYSTEM_PROMPT_THINKING_NATURAL_FILESYSTEM = (
    "You are acting inside AgentMemoryGym, a native WebShop bundled-shopping environment "
    "with a persistent workspace. "
    + _AGENTMEMORY_FILESYSTEM_REPLY_RULE_THINKING
    + _AGENTMEMORY_FILESYSTEM_ACTION_CONTRACT
    + _AGENTMEMORY_FILESYSTEM_MEMORY_GUIDANCE
)
AGENTMEMORY_ACTION_SYSTEM_PROMPT_REASONING_NATURAL_FILESYSTEM = (
    "You are acting inside AgentMemoryGym, a native WebShop bundled-shopping environment "
    "with a persistent workspace. "
    + _AGENTMEMORY_FILESYSTEM_REPLY_RULE_REASONING
    + _AGENTMEMORY_FILESYSTEM_ACTION_CONTRACT
    + _AGENTMEMORY_FILESYSTEM_MEMORY_GUIDANCE
)
AGENTMEMORY_ACTION_SYSTEM_PROMPT_RECENCY_FILESYSTEM = (
    "You are acting inside AgentMemoryGym, a native WebShop bundled-shopping environment "
    "with a persistent workspace. "
    + _AGENTMEMORY_FILESYSTEM_REPLY_RULE_NO_THINKING
    + _AGENTMEMORY_FILESYSTEM_ACTION_CONTRACT
    + _AGENTMEMORY_RECENCY_FILESYSTEM_MEMORY_GUIDANCE
)
AGENTMEMORY_ACTION_SYSTEM_PROMPT_THINKING_RECENCY_FILESYSTEM = (
    "You are acting inside AgentMemoryGym, a native WebShop bundled-shopping environment "
    "with a persistent workspace. "
    + _AGENTMEMORY_FILESYSTEM_REPLY_RULE_THINKING
    + _AGENTMEMORY_FILESYSTEM_ACTION_CONTRACT
    + _AGENTMEMORY_RECENCY_FILESYSTEM_MEMORY_GUIDANCE
)
AGENTMEMORY_ACTION_SYSTEM_PROMPT_REASONING_RECENCY_FILESYSTEM = (
    "You are acting inside AgentMemoryGym, a native WebShop bundled-shopping environment "
    "with a persistent workspace. "
    + _AGENTMEMORY_FILESYSTEM_REPLY_RULE_REASONING
    + _AGENTMEMORY_FILESYSTEM_ACTION_CONTRACT
    + _AGENTMEMORY_RECENCY_FILESYSTEM_MEMORY_GUIDANCE
)
AGENTMEMORY_ACTION_SYSTEM_PROMPT_DISTRACTOR_FILESYSTEM = (
    "You are acting inside AgentMemoryGym, a native WebShop bundled-shopping environment "
    "with a persistent workspace. "
    + _AGENTMEMORY_FILESYSTEM_REPLY_RULE_NO_THINKING
    + _AGENTMEMORY_FILESYSTEM_ACTION_CONTRACT
    + _AGENTMEMORY_DISTRACTOR_FILESYSTEM_MEMORY_GUIDANCE
)
AGENTMEMORY_ACTION_SYSTEM_PROMPT_THINKING_DISTRACTOR_FILESYSTEM = (
    "You are acting inside AgentMemoryGym, a native WebShop bundled-shopping environment "
    "with a persistent workspace. "
    + _AGENTMEMORY_FILESYSTEM_REPLY_RULE_THINKING
    + _AGENTMEMORY_FILESYSTEM_ACTION_CONTRACT
    + _AGENTMEMORY_DISTRACTOR_FILESYSTEM_MEMORY_GUIDANCE
)
AGENTMEMORY_ACTION_SYSTEM_PROMPT_REASONING_DISTRACTOR_FILESYSTEM = (
    "You are acting inside AgentMemoryGym, a native WebShop bundled-shopping environment "
    "with a persistent workspace. "
    + _AGENTMEMORY_FILESYSTEM_REPLY_RULE_REASONING
    + _AGENTMEMORY_FILESYSTEM_ACTION_CONTRACT
    + _AGENTMEMORY_DISTRACTOR_FILESYSTEM_MEMORY_GUIDANCE
)
AGENTMEMORY_ACTION_SYSTEM_PROMPT_LATENT_PREFERENCE_FILESYSTEM = (
    "You are acting inside AgentMemoryGym, a native WebShop bundled-shopping environment "
    "with a persistent workspace. "
    + _AGENTMEMORY_FILESYSTEM_REPLY_RULE_NO_THINKING
    + _AGENTMEMORY_FILESYSTEM_ACTION_CONTRACT
    + _AGENTMEMORY_LATENT_PREFERENCE_FILESYSTEM_MEMORY_GUIDANCE
)
AGENTMEMORY_ACTION_SYSTEM_PROMPT_THINKING_LATENT_PREFERENCE_FILESYSTEM = (
    "You are acting inside AgentMemoryGym, a native WebShop bundled-shopping environment "
    "with a persistent workspace. "
    + _AGENTMEMORY_FILESYSTEM_REPLY_RULE_THINKING
    + _AGENTMEMORY_FILESYSTEM_ACTION_CONTRACT
    + _AGENTMEMORY_LATENT_PREFERENCE_FILESYSTEM_MEMORY_GUIDANCE
)
AGENTMEMORY_ACTION_SYSTEM_PROMPT_REASONING_LATENT_PREFERENCE_FILESYSTEM = (
    "You are acting inside AgentMemoryGym, a native WebShop bundled-shopping environment "
    "with a persistent workspace. "
    + _AGENTMEMORY_FILESYSTEM_REPLY_RULE_REASONING
    + _AGENTMEMORY_FILESYSTEM_ACTION_CONTRACT
    + _AGENTMEMORY_LATENT_PREFERENCE_FILESYSTEM_MEMORY_GUIDANCE
)
AGENTMEMORY_ACTION_SYSTEM_PROMPT_COMPOSITIONAL_FILESYSTEM = (
    "You are acting inside AgentMemoryGym, a native WebShop bundled-shopping environment "
    "with a persistent workspace. "
    + _AGENTMEMORY_FILESYSTEM_REPLY_RULE_NO_THINKING
    + _AGENTMEMORY_FILESYSTEM_ACTION_CONTRACT
    + _AGENTMEMORY_COMPOSITIONAL_FILESYSTEM_MEMORY_GUIDANCE
)
AGENTMEMORY_ACTION_SYSTEM_PROMPT_THINKING_COMPOSITIONAL_FILESYSTEM = (
    "You are acting inside AgentMemoryGym, a native WebShop bundled-shopping environment "
    "with a persistent workspace. "
    + _AGENTMEMORY_FILESYSTEM_REPLY_RULE_THINKING
    + _AGENTMEMORY_FILESYSTEM_ACTION_CONTRACT
    + _AGENTMEMORY_COMPOSITIONAL_FILESYSTEM_MEMORY_GUIDANCE
)
AGENTMEMORY_ACTION_SYSTEM_PROMPT_REASONING_COMPOSITIONAL_FILESYSTEM = (
    "You are acting inside AgentMemoryGym, a native WebShop bundled-shopping environment "
    "with a persistent workspace. "
    + _AGENTMEMORY_FILESYSTEM_REPLY_RULE_REASONING
    + _AGENTMEMORY_FILESYSTEM_ACTION_CONTRACT
    + _AGENTMEMORY_COMPOSITIONAL_FILESYSTEM_MEMORY_GUIDANCE
)
AGENTMEMORY_ACTION_SYSTEM_PROMPT_NEGATIVE_FILESYSTEM = (
    "You are acting inside AgentMemoryGym, a native WebShop bundled-shopping environment "
    "with a persistent workspace. "
    + _AGENTMEMORY_FILESYSTEM_REPLY_RULE_NO_THINKING
    + _AGENTMEMORY_FILESYSTEM_ACTION_CONTRACT
    + _AGENTMEMORY_NEGATIVE_FILESYSTEM_MEMORY_GUIDANCE
)
AGENTMEMORY_ACTION_SYSTEM_PROMPT_THINKING_NEGATIVE_FILESYSTEM = (
    "You are acting inside AgentMemoryGym, a native WebShop bundled-shopping environment "
    "with a persistent workspace. "
    + _AGENTMEMORY_FILESYSTEM_REPLY_RULE_THINKING
    + _AGENTMEMORY_FILESYSTEM_ACTION_CONTRACT
    + _AGENTMEMORY_NEGATIVE_FILESYSTEM_MEMORY_GUIDANCE
)
AGENTMEMORY_ACTION_SYSTEM_PROMPT_REASONING_NEGATIVE_FILESYSTEM = (
    "You are acting inside AgentMemoryGym, a native WebShop bundled-shopping environment "
    "with a persistent workspace. "
    + _AGENTMEMORY_FILESYSTEM_REPLY_RULE_REASONING
    + _AGENTMEMORY_FILESYSTEM_ACTION_CONTRACT
    + _AGENTMEMORY_NEGATIVE_FILESYSTEM_MEMORY_GUIDANCE
)
AGENTMEMORY_ACTION_SYSTEM_PROMPT_INTENT_CLARIFICATION_FILESYSTEM = (
    "You are acting inside AgentMemoryGym, a native WebShop bundled-shopping environment "
    "with a persistent workspace. "
    + _AGENTMEMORY_FILESYSTEM_REPLY_RULE_NO_THINKING
    + _AGENTMEMORY_FILESYSTEM_ACTION_CONTRACT
    + _AGENTMEMORY_INTENT_CLARIFICATION_FILESYSTEM_MEMORY_GUIDANCE
)
AGENTMEMORY_ACTION_SYSTEM_PROMPT_THINKING_INTENT_CLARIFICATION_FILESYSTEM = (
    "You are acting inside AgentMemoryGym, a native WebShop bundled-shopping environment "
    "with a persistent workspace. "
    + _AGENTMEMORY_FILESYSTEM_REPLY_RULE_THINKING
    + _AGENTMEMORY_FILESYSTEM_ACTION_CONTRACT
    + _AGENTMEMORY_INTENT_CLARIFICATION_FILESYSTEM_MEMORY_GUIDANCE
)
AGENTMEMORY_ACTION_SYSTEM_PROMPT_REASONING_INTENT_CLARIFICATION_FILESYSTEM = (
    "You are acting inside AgentMemoryGym, a native WebShop bundled-shopping environment "
    "with a persistent workspace. "
    + _AGENTMEMORY_FILESYSTEM_REPLY_RULE_REASONING
    + _AGENTMEMORY_FILESYSTEM_ACTION_CONTRACT
    + _AGENTMEMORY_INTENT_CLARIFICATION_FILESYSTEM_MEMORY_GUIDANCE
)
AGENTMEMORY_ACTION_SYSTEM_PROMPT_SELECTIVE_MEMORY_USE_FILESYSTEM = (
    "You are acting inside AgentMemoryGym, a native WebShop bundled-shopping environment "
    "with a persistent workspace. "
    + _AGENTMEMORY_FILESYSTEM_REPLY_RULE_NO_THINKING
    + _AGENTMEMORY_FILESYSTEM_ACTION_CONTRACT
    + _AGENTMEMORY_SELECTIVE_MEMORY_USE_FILESYSTEM_MEMORY_GUIDANCE
)
AGENTMEMORY_ACTION_SYSTEM_PROMPT_THINKING_SELECTIVE_MEMORY_USE_FILESYSTEM = (
    "You are acting inside AgentMemoryGym, a native WebShop bundled-shopping environment "
    "with a persistent workspace. "
    + _AGENTMEMORY_FILESYSTEM_REPLY_RULE_THINKING
    + _AGENTMEMORY_FILESYSTEM_ACTION_CONTRACT
    + _AGENTMEMORY_SELECTIVE_MEMORY_USE_FILESYSTEM_MEMORY_GUIDANCE
)
AGENTMEMORY_ACTION_SYSTEM_PROMPT_REASONING_SELECTIVE_MEMORY_USE_FILESYSTEM = (
    "You are acting inside AgentMemoryGym, a native WebShop bundled-shopping environment "
    "with a persistent workspace. "
    + _AGENTMEMORY_FILESYSTEM_REPLY_RULE_REASONING
    + _AGENTMEMORY_FILESYSTEM_ACTION_CONTRACT
    + _AGENTMEMORY_SELECTIVE_MEMORY_USE_FILESYSTEM_MEMORY_GUIDANCE
)
# Keep the short names used by the evidence/evaluation loader as canonical
# aliases; the longer names above make the task family explicit in this file.
AGENTMEMORY_ACTION_SYSTEM_PROMPT_LATENT_FILESYSTEM = (
    AGENTMEMORY_ACTION_SYSTEM_PROMPT_LATENT_PREFERENCE_FILESYSTEM
)
AGENTMEMORY_ACTION_SYSTEM_PROMPT_THINKING_LATENT_FILESYSTEM = (
    AGENTMEMORY_ACTION_SYSTEM_PROMPT_THINKING_LATENT_PREFERENCE_FILESYSTEM
)
AGENTMEMORY_ACTION_SYSTEM_PROMPT_REASONING_LATENT_FILESYSTEM = (
    AGENTMEMORY_ACTION_SYSTEM_PROMPT_REASONING_LATENT_PREFERENCE_FILESYSTEM
)
AGENTMEMORY_ACTION_SYSTEM_PROMPT_INTENT_FILESYSTEM = (
    AGENTMEMORY_ACTION_SYSTEM_PROMPT_INTENT_CLARIFICATION_FILESYSTEM
)
AGENTMEMORY_ACTION_SYSTEM_PROMPT_THINKING_INTENT_FILESYSTEM = (
    AGENTMEMORY_ACTION_SYSTEM_PROMPT_THINKING_INTENT_CLARIFICATION_FILESYSTEM
)
AGENTMEMORY_ACTION_SYSTEM_PROMPT_REASONING_INTENT_FILESYSTEM = (
    AGENTMEMORY_ACTION_SYSTEM_PROMPT_REASONING_INTENT_CLARIFICATION_FILESYSTEM
)
AGENTMEMORY_ACTION_SYSTEM_PROMPT_SELECTIVE_FILESYSTEM = (
    AGENTMEMORY_ACTION_SYSTEM_PROMPT_SELECTIVE_MEMORY_USE_FILESYSTEM
)
AGENTMEMORY_ACTION_SYSTEM_PROMPT_THINKING_SELECTIVE_FILESYSTEM = (
    AGENTMEMORY_ACTION_SYSTEM_PROMPT_THINKING_SELECTIVE_MEMORY_USE_FILESYSTEM
)
AGENTMEMORY_ACTION_SYSTEM_PROMPT_REASONING_SELECTIVE_FILESYSTEM = (
    AGENTMEMORY_ACTION_SYSTEM_PROMPT_REASONING_SELECTIVE_MEMORY_USE_FILESYSTEM
)
AGENTMEMORY_ACTION_SYSTEM_PROMPT_NO_WORKSPACE = (
    "You are acting inside AgentMemoryGym, a native WebShop bundled-shopping environment "
    "without a persistent workspace. "
    + _AGENTMEMORY_NO_WORKSPACE_REPLY_RULE_NO_THINKING
    + _AGENTMEMORY_NO_WORKSPACE_ACTION_CONTRACT
)
AGENTMEMORY_ACTION_SYSTEM_PROMPT_THINKING_NO_WORKSPACE = (
    "You are acting inside AgentMemoryGym, a native WebShop bundled-shopping environment "
    "without a persistent workspace. "
    + _AGENTMEMORY_NO_WORKSPACE_REPLY_RULE_THINKING
    + _AGENTMEMORY_NO_WORKSPACE_ACTION_CONTRACT
)
AGENTMEMORY_ACTION_SYSTEM_PROMPT_REASONING_NO_WORKSPACE = (
    "You are acting inside AgentMemoryGym, a native WebShop bundled-shopping environment "
    "without a persistent workspace. "
    + _AGENTMEMORY_NO_WORKSPACE_REPLY_RULE_REASONING
    + _AGENTMEMORY_NO_WORKSPACE_ACTION_CONTRACT
)

AGENTMEMORY_QUERY_TOP1_SURFACES = frozenset(
    {
        "agentmemory_webshop_distractor_robustness_top1_train_v1",
        "agentmemory_webshop_compositional_recall_top1_train_v1",
        "agentmemory_webshop_intent_clarification_train_v1",
        "agentmemory_webshop_selective_memory_use_top1_train_v1",
        "agentmemory_webshop_negative_constraint_top1_train_v1",
    }
)
AGENTMEMORY_INTENT_CLARIFICATION_SURFACE = (
    "agentmemory_webshop_intent_clarification_train_v1"
)
_AGENTMEMORY_QUERY_TOP1_RETRIEVAL_CONTRACT = (
    "RETRIEVE requires exactly query:string and returns exactly "
    "one highest-ranked matching memory. memory_id and top_k are forbidden."
)
_AGENTMEMORY_DEFAULT_RETRIEVAL_CONTRACT = (
    "RETRIEVE accepts exactly one lookup field: query:string for BM25 text matching "
    "with optional top_k:int (default 3), or memory_id:string for exact readback of "
    "that entry."
)
_AGENTMEMORY_INTENT_CLARIFICATION_CONTRACT = (
    "In the first shopping session, the request is intentionally ambiguous. ASK "
    "requires field:string and may be used exactly once with the ambiguity-resolving "
    "field named by the task. The environment returns a CLARIFY observation; store "
    "that clarification before the first purchase and retrieve it in later sessions."
)

_AGENTMEMORY_LTM_KEY_INVENTORY_CONTRACT = (
    "The observation includes a key-only long-term memory inventory when that interface "
    "variant is enabled. It lists only the memory_id and a policy-authored lookup key; memory "
    "values remain hidden until RETRIEVE. In this variant each key must be a single ASCII "
    "lookup label of at most 24 letters, digits, spaces, underscores, or hyphens, without a "
    "leading or trailing separator; put product identity and compatibility facts in value, "
    "not key. RETRIEVE matches "
    "both the key and value of memories previously written with ADD."
)

# Full prompts for each mode: same intro and action contract, different reply rule.
AGENTMEMORY_ACTION_SYSTEM_PROMPT = (
    _AGENTMEMORY_INTRO
    + _AGENTMEMORY_REPLY_RULE_NO_THINKING
    + _AGENTMEMORY_ACTION_CONTRACT
    + " "
    + _AGENTMEMORY_MEMORY_LIFECYCLE
)
AGENTMEMORY_ACTION_SYSTEM_PROMPT_THINKING = (
    _AGENTMEMORY_INTRO
    + _AGENTMEMORY_REPLY_RULE_THINKING
    + _AGENTMEMORY_ACTION_CONTRACT
    + " "
    + _AGENTMEMORY_MEMORY_LIFECYCLE
)
AGENTMEMORY_ACTION_SYSTEM_PROMPT_REASONING = (
    _AGENTMEMORY_INTRO
    + _AGENTMEMORY_REPLY_RULE_REASONING
    + _AGENTMEMORY_ACTION_CONTRACT
    + " "
    + _AGENTMEMORY_MEMORY_LIFECYCLE
)
AGENTMEMORY_ACTION_SYSTEM_PROMPT_NEUTRAL = (
    _AGENTMEMORY_INTRO
    + _AGENTMEMORY_REPLY_RULE_NO_THINKING
    + _AGENTMEMORY_ACTION_CONTRACT
)
AGENTMEMORY_ACTION_SYSTEM_PROMPT_THINKING_NEUTRAL = (
    _AGENTMEMORY_INTRO
    + _AGENTMEMORY_REPLY_RULE_THINKING
    + _AGENTMEMORY_ACTION_CONTRACT
)
AGENTMEMORY_ACTION_SYSTEM_PROMPT_REASONING_NEUTRAL = (
    _AGENTMEMORY_INTRO
    + _AGENTMEMORY_REPLY_RULE_REASONING
    + _AGENTMEMORY_ACTION_CONTRACT
)
AGENTMEMORY_ACTION_SYSTEM_PROMPT_NEUTRAL_HORIZON = (
    AGENTMEMORY_ACTION_SYSTEM_PROMPT_NEUTRAL
    + " "
    + _AGENTMEMORY_TASK_HORIZON
)
AGENTMEMORY_ACTION_SYSTEM_PROMPT_THINKING_NEUTRAL_HORIZON = (
    AGENTMEMORY_ACTION_SYSTEM_PROMPT_THINKING_NEUTRAL
    + " "
    + _AGENTMEMORY_TASK_HORIZON
)
AGENTMEMORY_ACTION_SYSTEM_PROMPT_REASONING_NEUTRAL_HORIZON = (
    AGENTMEMORY_ACTION_SYSTEM_PROMPT_REASONING_NEUTRAL
    + " "
    + _AGENTMEMORY_TASK_HORIZON
)
AGENTMEMORY_ACTION_SYSTEM_PROMPT_NEUTRAL_HORIZON_RESPONSIBILITY = (
    AGENTMEMORY_ACTION_SYSTEM_PROMPT_NEUTRAL_HORIZON
    + " "
    + _AGENTMEMORY_CROSS_SESSION_MEMORY_RESPONSIBILITY
)
AGENTMEMORY_ACTION_SYSTEM_PROMPT_LATENT_PREFERENCE_SOP = (
    AGENTMEMORY_ACTION_SYSTEM_PROMPT_NEUTRAL
    + " "
    + _AGENTMEMORY_LATENT_PREFERENCE_SOP
)
AGENTMEMORY_ACTION_SYSTEM_PROMPT_THINKING_LATENT_PREFERENCE_SOP = (
    AGENTMEMORY_ACTION_SYSTEM_PROMPT_THINKING_NEUTRAL
    + " "
    + _AGENTMEMORY_LATENT_PREFERENCE_SOP
)
AGENTMEMORY_ACTION_SYSTEM_PROMPT_REASONING_LATENT_PREFERENCE_SOP = (
    AGENTMEMORY_ACTION_SYSTEM_PROMPT_REASONING_NEUTRAL
    + " "
    + _AGENTMEMORY_LATENT_PREFERENCE_SOP
)
AGENTMEMORY_ACTION_SYSTEM_PROMPT_THINKING_NEUTRAL_HORIZON_RESPONSIBILITY = (
    AGENTMEMORY_ACTION_SYSTEM_PROMPT_THINKING_NEUTRAL_HORIZON
    + " "
    + _AGENTMEMORY_CROSS_SESSION_MEMORY_RESPONSIBILITY
)
AGENTMEMORY_ACTION_SYSTEM_PROMPT_REASONING_NEUTRAL_HORIZON_RESPONSIBILITY = (
    AGENTMEMORY_ACTION_SYSTEM_PROMPT_REASONING_NEUTRAL_HORIZON
    + " "
    + _AGENTMEMORY_CROSS_SESSION_MEMORY_RESPONSIBILITY
)
AGENTMEMORY_ACTION_SYSTEM_PROMPT_LTM_KEY_INVENTORY = (
    AGENTMEMORY_ACTION_SYSTEM_PROMPT
    + " "
    + _AGENTMEMORY_LTM_KEY_INVENTORY_CONTRACT
)
AGENTMEMORY_ACTION_SYSTEM_PROMPT_THINKING_LTM_KEY_INVENTORY = (
    AGENTMEMORY_ACTION_SYSTEM_PROMPT_THINKING
    + " "
    + _AGENTMEMORY_LTM_KEY_INVENTORY_CONTRACT
)
AGENTMEMORY_ACTION_SYSTEM_PROMPT_REASONING_LTM_KEY_INVENTORY = (
    AGENTMEMORY_ACTION_SYSTEM_PROMPT_REASONING
    + " "
    + _AGENTMEMORY_LTM_KEY_INVENTORY_CONTRACT
)
AGENTMEMORY_ACTION_SYSTEM_PROMPT_NEUTRAL_LTM_KEY_INVENTORY = (
    AGENTMEMORY_ACTION_SYSTEM_PROMPT_NEUTRAL
    + " "
    + _AGENTMEMORY_LTM_KEY_INVENTORY_CONTRACT
)
AGENTMEMORY_ACTION_SYSTEM_PROMPT_THINKING_NEUTRAL_LTM_KEY_INVENTORY = (
    AGENTMEMORY_ACTION_SYSTEM_PROMPT_THINKING_NEUTRAL
    + " "
    + _AGENTMEMORY_LTM_KEY_INVENTORY_CONTRACT
)
AGENTMEMORY_ACTION_SYSTEM_PROMPT_REASONING_NEUTRAL_LTM_KEY_INVENTORY = (
    AGENTMEMORY_ACTION_SYSTEM_PROMPT_REASONING_NEUTRAL
    + " "
    + _AGENTMEMORY_LTM_KEY_INVENTORY_CONTRACT
)
AGENTMEMORY_ACTION_SYSTEM_PROMPT_NEUTRAL_HORIZON_LTM_KEY_INVENTORY = (
    AGENTMEMORY_ACTION_SYSTEM_PROMPT_NEUTRAL_HORIZON
    + " "
    + _AGENTMEMORY_LTM_KEY_INVENTORY_CONTRACT
)
AGENTMEMORY_ACTION_SYSTEM_PROMPT_THINKING_NEUTRAL_HORIZON_LTM_KEY_INVENTORY = (
    AGENTMEMORY_ACTION_SYSTEM_PROMPT_THINKING_NEUTRAL_HORIZON
    + " "
    + _AGENTMEMORY_LTM_KEY_INVENTORY_CONTRACT
)
AGENTMEMORY_ACTION_SYSTEM_PROMPT_REASONING_NEUTRAL_HORIZON_LTM_KEY_INVENTORY = (
    AGENTMEMORY_ACTION_SYSTEM_PROMPT_REASONING_NEUTRAL_HORIZON
    + " "
    + _AGENTMEMORY_LTM_KEY_INVENTORY_CONTRACT
)
AGENTMEMORY_ACTION_SYSTEM_PROMPT_NEUTRAL_HORIZON_RESPONSIBILITY_LTM_KEY_INVENTORY = (
    AGENTMEMORY_ACTION_SYSTEM_PROMPT_NEUTRAL_HORIZON_RESPONSIBILITY
    + " "
    + _AGENTMEMORY_LTM_KEY_INVENTORY_CONTRACT
)
AGENTMEMORY_ACTION_SYSTEM_PROMPT_LATENT_PREFERENCE_SOP_LTM_KEY_INVENTORY = (
    AGENTMEMORY_ACTION_SYSTEM_PROMPT_LATENT_PREFERENCE_SOP
    + " "
    + _AGENTMEMORY_LTM_KEY_INVENTORY_CONTRACT
)
AGENTMEMORY_ACTION_SYSTEM_PROMPT_THINKING_LATENT_PREFERENCE_SOP_LTM_KEY_INVENTORY = (
    AGENTMEMORY_ACTION_SYSTEM_PROMPT_THINKING_LATENT_PREFERENCE_SOP
    + " "
    + _AGENTMEMORY_LTM_KEY_INVENTORY_CONTRACT
)
AGENTMEMORY_ACTION_SYSTEM_PROMPT_REASONING_LATENT_PREFERENCE_SOP_LTM_KEY_INVENTORY = (
    AGENTMEMORY_ACTION_SYSTEM_PROMPT_REASONING_LATENT_PREFERENCE_SOP
    + " "
    + _AGENTMEMORY_LTM_KEY_INVENTORY_CONTRACT
)
AGENTMEMORY_ACTION_SYSTEM_PROMPT_THINKING_NEUTRAL_HORIZON_RESPONSIBILITY_LTM_KEY_INVENTORY = (
    AGENTMEMORY_ACTION_SYSTEM_PROMPT_THINKING_NEUTRAL_HORIZON_RESPONSIBILITY
    + " "
    + _AGENTMEMORY_LTM_KEY_INVENTORY_CONTRACT
)
AGENTMEMORY_ACTION_SYSTEM_PROMPT_REASONING_NEUTRAL_HORIZON_RESPONSIBILITY_LTM_KEY_INVENTORY = (
    AGENTMEMORY_ACTION_SYSTEM_PROMPT_REASONING_NEUTRAL_HORIZON_RESPONSIBILITY
    + " "
    + _AGENTMEMORY_LTM_KEY_INVENTORY_CONTRACT
)


def agentmemory_action_system_prompt(
    *,
    ltm_inventory_mode: str | None = None,
    memory_prompt_mode: str | None = None,
    surface: str | None = None,
    workspace_enabled: bool = True,
) -> str:
    # Pick the reply rule that matches the active thinking mode so the prompt
    # never contradicts what the chat template does with <think>.
    inventory_mode = (
        agentmemory_ltm_inventory_mode()
        if ltm_inventory_mode is None
        else ltm_inventory_mode
    )
    if inventory_mode not in ("hidden", "keys"):
        raise ValueError("ltm_inventory_mode must be 'hidden' or 'keys'.")
    prompt_mode = (
        agentmemory_memory_prompt_mode()
        if memory_prompt_mode is None
        else memory_prompt_mode
    )
    if prompt_mode not in AGENTMEMORY_MEMORY_PROMPT_MODES:
        raise ValueError(
            "memory_prompt_mode must be one of: "
            + ", ".join(AGENTMEMORY_MEMORY_PROMPT_MODES)
            + "."
        )
    if type(workspace_enabled) is not bool:
        raise ValueError("workspace_enabled must be boolean.")
    if not workspace_enabled and prompt_mode != NATURAL_FILESYSTEM_PROMPT_MODE:
        raise ValueError(
            "workspace_enabled=False is valid only for natural_filesystem."
        )
    if prompt_mode == NATURAL_FILESYSTEM_PROMPT_MODE:
        if inventory_mode != "hidden":
            raise ValueError(
                "natural_filesystem requires ltm_inventory_mode='hidden'."
            )
        if surface is not None and surface not in FILESYSTEM_SURFACES:
            raise ValueError(
                "natural_filesystem is only valid for the persistent-workspace "
                "surfaces: " + ", ".join(sorted(FILESYSTEM_SURFACES)) + "."
            )
        if not workspace_enabled:
            if _agentmemory_thinking_enabled():
                return AGENTMEMORY_ACTION_SYSTEM_PROMPT_THINKING_NO_WORKSPACE
            if _agentmemory_reasoning_enabled():
                return AGENTMEMORY_ACTION_SYSTEM_PROMPT_REASONING_NO_WORKSPACE
            return AGENTMEMORY_ACTION_SYSTEM_PROMPT_NO_WORKSPACE
        effective_surface = surface or NATURAL_FILESYSTEM_SURFACE
        prompt_triplet = {
            NATURAL_FILESYSTEM_SURFACE: (
                AGENTMEMORY_ACTION_SYSTEM_PROMPT_NATURAL_FILESYSTEM,
                AGENTMEMORY_ACTION_SYSTEM_PROMPT_THINKING_NATURAL_FILESYSTEM,
                AGENTMEMORY_ACTION_SYSTEM_PROMPT_REASONING_NATURAL_FILESYSTEM,
            ),
            RECENCY_OVERRIDE_FILESYSTEM_SURFACE: (
                AGENTMEMORY_ACTION_SYSTEM_PROMPT_RECENCY_FILESYSTEM,
                AGENTMEMORY_ACTION_SYSTEM_PROMPT_THINKING_RECENCY_FILESYSTEM,
                AGENTMEMORY_ACTION_SYSTEM_PROMPT_REASONING_RECENCY_FILESYSTEM,
            ),
            LATENT_PREFERENCE_FILESYSTEM_SURFACE: (
                AGENTMEMORY_ACTION_SYSTEM_PROMPT_LATENT_PREFERENCE_FILESYSTEM,
                AGENTMEMORY_ACTION_SYSTEM_PROMPT_THINKING_LATENT_PREFERENCE_FILESYSTEM,
                AGENTMEMORY_ACTION_SYSTEM_PROMPT_REASONING_LATENT_PREFERENCE_FILESYSTEM,
            ),
            DISTRACTOR_ROBUSTNESS_FILESYSTEM_SURFACE: (
                AGENTMEMORY_ACTION_SYSTEM_PROMPT_DISTRACTOR_FILESYSTEM,
                AGENTMEMORY_ACTION_SYSTEM_PROMPT_THINKING_DISTRACTOR_FILESYSTEM,
                AGENTMEMORY_ACTION_SYSTEM_PROMPT_REASONING_DISTRACTOR_FILESYSTEM,
            ),
            COMPOSITIONAL_RECALL_FILESYSTEM_SURFACE: (
                AGENTMEMORY_ACTION_SYSTEM_PROMPT_COMPOSITIONAL_FILESYSTEM,
                AGENTMEMORY_ACTION_SYSTEM_PROMPT_THINKING_COMPOSITIONAL_FILESYSTEM,
                AGENTMEMORY_ACTION_SYSTEM_PROMPT_REASONING_COMPOSITIONAL_FILESYSTEM,
            ),
            NEGATIVE_CONSTRAINT_FILESYSTEM_SURFACE: (
                AGENTMEMORY_ACTION_SYSTEM_PROMPT_NEGATIVE_FILESYSTEM,
                AGENTMEMORY_ACTION_SYSTEM_PROMPT_THINKING_NEGATIVE_FILESYSTEM,
                AGENTMEMORY_ACTION_SYSTEM_PROMPT_REASONING_NEGATIVE_FILESYSTEM,
            ),
            INTENT_CLARIFICATION_FILESYSTEM_SURFACE: (
                AGENTMEMORY_ACTION_SYSTEM_PROMPT_INTENT_CLARIFICATION_FILESYSTEM,
                AGENTMEMORY_ACTION_SYSTEM_PROMPT_THINKING_INTENT_CLARIFICATION_FILESYSTEM,
                AGENTMEMORY_ACTION_SYSTEM_PROMPT_REASONING_INTENT_CLARIFICATION_FILESYSTEM,
            ),
            SELECTIVE_MEMORY_USE_FILESYSTEM_SURFACE: (
                AGENTMEMORY_ACTION_SYSTEM_PROMPT_SELECTIVE_MEMORY_USE_FILESYSTEM,
                AGENTMEMORY_ACTION_SYSTEM_PROMPT_THINKING_SELECTIVE_MEMORY_USE_FILESYSTEM,
                AGENTMEMORY_ACTION_SYSTEM_PROMPT_REASONING_SELECTIVE_MEMORY_USE_FILESYSTEM,
            ),
        }[effective_surface]
        if _agentmemory_thinking_enabled():
            return prompt_triplet[1]
        if _agentmemory_reasoning_enabled():
            return prompt_triplet[2]
        return prompt_triplet[0]
    key_inventory = inventory_mode == "keys"
    latent_preference = prompt_mode == "latent_preference_sop"
    selective_memory = prompt_mode == "selective_memory_sop"
    neutral = prompt_mode in (
        "neutral",
        "neutral_horizon",
        "neutral_horizon_responsibility",
    ) or selective_memory
    neutral_horizon = prompt_mode in (
        "neutral_horizon",
        "neutral_horizon_responsibility",
    )
    responsibility = prompt_mode == "neutral_horizon_responsibility"

    def finish(prompt: str) -> str:
        if surface in AGENTMEMORY_QUERY_TOP1_SURFACES:
            if prompt.count(_AGENTMEMORY_DEFAULT_RETRIEVAL_CONTRACT) != 1:
                raise RuntimeError(
                    "AgentMemory query-top1 prompt could not replace the default "
                    "RETRIEVE contract exactly once."
                )
            prompt = prompt.replace(
                _AGENTMEMORY_DEFAULT_RETRIEVAL_CONTRACT,
                _AGENTMEMORY_QUERY_TOP1_RETRIEVAL_CONTRACT,
                1,
            )
        if surface == AGENTMEMORY_INTENT_CLARIFICATION_SURFACE:
            prompt += " " + _AGENTMEMORY_INTENT_CLARIFICATION_CONTRACT
        if selective_memory:
            prompt += " " + _AGENTMEMORY_SELECTIVE_MEMORY_SOP
        return prompt

    if _agentmemory_thinking_enabled():
        if key_inventory:
            if latent_preference:
                return finish(AGENTMEMORY_ACTION_SYSTEM_PROMPT_THINKING_LATENT_PREFERENCE_SOP_LTM_KEY_INVENTORY)
            if responsibility:
                return finish(AGENTMEMORY_ACTION_SYSTEM_PROMPT_THINKING_NEUTRAL_HORIZON_RESPONSIBILITY_LTM_KEY_INVENTORY)
            if neutral_horizon:
                return finish(AGENTMEMORY_ACTION_SYSTEM_PROMPT_THINKING_NEUTRAL_HORIZON_LTM_KEY_INVENTORY)
            if neutral:
                return finish(AGENTMEMORY_ACTION_SYSTEM_PROMPT_THINKING_NEUTRAL_LTM_KEY_INVENTORY)
            return finish(AGENTMEMORY_ACTION_SYSTEM_PROMPT_THINKING_LTM_KEY_INVENTORY)
        if latent_preference:
            return finish(AGENTMEMORY_ACTION_SYSTEM_PROMPT_THINKING_LATENT_PREFERENCE_SOP)
        if responsibility:
            return finish(AGENTMEMORY_ACTION_SYSTEM_PROMPT_THINKING_NEUTRAL_HORIZON_RESPONSIBILITY)
        if neutral_horizon:
            return finish(AGENTMEMORY_ACTION_SYSTEM_PROMPT_THINKING_NEUTRAL_HORIZON)
        if neutral:
            return finish(AGENTMEMORY_ACTION_SYSTEM_PROMPT_THINKING_NEUTRAL)
        return finish(AGENTMEMORY_ACTION_SYSTEM_PROMPT_THINKING)
    if _agentmemory_reasoning_enabled():
        if key_inventory:
            if latent_preference:
                return finish(AGENTMEMORY_ACTION_SYSTEM_PROMPT_REASONING_LATENT_PREFERENCE_SOP_LTM_KEY_INVENTORY)
            if responsibility:
                return finish(AGENTMEMORY_ACTION_SYSTEM_PROMPT_REASONING_NEUTRAL_HORIZON_RESPONSIBILITY_LTM_KEY_INVENTORY)
            if neutral_horizon:
                return finish(AGENTMEMORY_ACTION_SYSTEM_PROMPT_REASONING_NEUTRAL_HORIZON_LTM_KEY_INVENTORY)
            if neutral:
                return finish(AGENTMEMORY_ACTION_SYSTEM_PROMPT_REASONING_NEUTRAL_LTM_KEY_INVENTORY)
            return finish(AGENTMEMORY_ACTION_SYSTEM_PROMPT_REASONING_LTM_KEY_INVENTORY)
        if latent_preference:
            return finish(AGENTMEMORY_ACTION_SYSTEM_PROMPT_REASONING_LATENT_PREFERENCE_SOP)
        if responsibility:
            return finish(AGENTMEMORY_ACTION_SYSTEM_PROMPT_REASONING_NEUTRAL_HORIZON_RESPONSIBILITY)
        if neutral_horizon:
            return finish(AGENTMEMORY_ACTION_SYSTEM_PROMPT_REASONING_NEUTRAL_HORIZON)
        if neutral:
            return finish(AGENTMEMORY_ACTION_SYSTEM_PROMPT_REASONING_NEUTRAL)
        return finish(AGENTMEMORY_ACTION_SYSTEM_PROMPT_REASONING)
    if key_inventory:
        if latent_preference:
            return finish(AGENTMEMORY_ACTION_SYSTEM_PROMPT_LATENT_PREFERENCE_SOP_LTM_KEY_INVENTORY)
        if responsibility:
            return finish(AGENTMEMORY_ACTION_SYSTEM_PROMPT_NEUTRAL_HORIZON_RESPONSIBILITY_LTM_KEY_INVENTORY)
        if neutral_horizon:
            return finish(AGENTMEMORY_ACTION_SYSTEM_PROMPT_NEUTRAL_HORIZON_LTM_KEY_INVENTORY)
        if neutral:
            return finish(AGENTMEMORY_ACTION_SYSTEM_PROMPT_NEUTRAL_LTM_KEY_INVENTORY)
        return finish(AGENTMEMORY_ACTION_SYSTEM_PROMPT_LTM_KEY_INVENTORY)
    if latent_preference:
        return finish(AGENTMEMORY_ACTION_SYSTEM_PROMPT_LATENT_PREFERENCE_SOP)
    if responsibility:
        return finish(AGENTMEMORY_ACTION_SYSTEM_PROMPT_NEUTRAL_HORIZON_RESPONSIBILITY)
    if neutral_horizon:
        return finish(AGENTMEMORY_ACTION_SYSTEM_PROMPT_NEUTRAL_HORIZON)
    if neutral:
        return finish(AGENTMEMORY_ACTION_SYSTEM_PROMPT_NEUTRAL)
    return finish(AGENTMEMORY_ACTION_SYSTEM_PROMPT)

def _normalize_chat_template_token_ids(encoded) -> List[int]:
    """Normalize tokenizer.apply_chat_template(..., tokenize=True) output.

    Some tokenizer implementations (observed with Qwen3.5-4B backed by
    Qwen2Tokenizer) return a BatchEncoding/dict rather than a plain list of
    token ids. Iterating that object yields string keys like "input_ids",
    which later breaks vLLM prompt validation.
    """
    if isinstance(encoded, dict) or (hasattr(encoded, "__getitem__") and "input_ids" in encoded):
        encoded = encoded["input_ids"]
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    # A batched tokenizer output may be [[...]] for a single conversation.
    if encoded and isinstance(encoded[0], (list, tuple)):
        if len(encoded) != 1:
            raise ValueError(f"Expected one chat-template sequence, got batch size {len(encoded)}")
        encoded = encoded[0]
    token_ids = list(encoded)
    bad = [(i, type(x).__name__, repr(x)[:80]) for i, x in enumerate(token_ids) if not isinstance(x, int)]
    if bad:
        raise TypeError(f"Chat template produced non-integer token ids: {bad[:5]}")
    return token_ids


def apply_chat_template(tokenizer: PreTrainedTokenizer, conversations: list[dict[str, str]]) -> List[int]:
    """Tokenize a conversation into generation-prompt token ids.

    enable_thinking follows AGENTMEMORY_ENABLE_THINKING: when off (default) the
    template closes the assistant turn with an empty <think></think> so the model
    emits a bare action; when on, the template leaves the <think> block open so
    the model can reason before acting. Qwen3.5 tokenizers that do not accept the
    enable_thinking kwarg fall back to the plain call (unchanged behaviour).
    """
    enable_thinking = _agentmemory_thinking_enabled()
    try:
        encoded = tokenizer.apply_chat_template(
            conversations,
            add_generation_prompt=True,
            tokenize=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        encoded = tokenizer.apply_chat_template(conversations, add_generation_prompt=True, tokenize=True)
    return _normalize_chat_template_token_ids(encoded)


# Backward-compatible alias: existing callers/imports of the old name keep working
# and now honour the AGENTMEMORY_ENABLE_THINKING flag through the shared function.
def apply_chat_template_no_thinking(tokenizer: PreTrainedTokenizer, conversations: list[dict[str, str]]) -> List[int]:
    return apply_chat_template(tokenizer, conversations)



def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids

class Message:
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content
    def to_dict(self):
        return {'role': self.role, 'content': self.content}
    def __repr__(self):
        return str(self.to_dict())
    def __str__(self):
        return self.__repr_

class RolloutHandler:
    def __init__(
        self,
        messages: List[Message],
        task_name: str,
        item_id: str | int,
        score: float,
        done: bool,
        input_ids: List[int],
        prompt_ids: List[int],
        response_ids: List[int],
        attention_mask: List[int],
        prompt_attention_mask: List[int],
        response_attention_mask: List[int],
        position_ids: List[int],
        prompt_position_ids: List[int],
        response_position_ids: List[int],
        loss_mask: List[int],
        prompt_loss_mask: List[int],
        response_loss_mask: List[int],
        max_response_len: int = 8192,
        max_model_len: int = 32768   
    ):
        self.messages = messages
        self.task_name = task_name
        self.item_id = item_id
        self.score = score
        self.done = done
        self.input_ids = input_ids
        self.prompt_ids = prompt_ids
        self.response_ids = response_ids
        self.attention_mask = attention_mask
        self.prompt_attention_mask = prompt_attention_mask
        self.response_attention_mask = response_attention_mask
        self.position_ids = position_ids
        self.prompt_position_ids = prompt_position_ids
        self.response_position_ids = response_position_ids
        self.loss_mask = loss_mask
        self.prompt_loss_mask = prompt_loss_mask
        self.response_loss_mask = response_loss_mask
        self.max_response_len = max_response_len
        self.max_model_len = max_model_len  
        self.format_config: dict = {
            "qwen": {
                "assistat_prefix_msg": "\n<|im_start|>assistant\n",
                "assistat_suffix_msg": "<|im_end|>",
                "user_prefix_msg": "\n<|im_start|>user\n",
                "user_suffix_msg": "<|im_end|>",
            }
        }

    def get_generation_prompt(self, tokenizer: PreTrainedTokenizer) -> List[int]:
        conversations = [
            msg.to_dict() for msg in self.messages
        ]
        return apply_chat_template(tokenizer, conversations)

    def get_latest_observation_prompt(
        self,
        tokenizer: PreTrainedTokenizer,
        *,
        system_prompt: str | None = None,
    ) -> List[int]:
        assert self.messages, "RolloutHandler has no messages."
        latest_user_message = self.messages[-1]
        assert latest_user_message.role == "user", (
            f"Latest-observation rollout expects the last message to be a user "
            f"observation, got role={latest_user_message.role!r}."
        )
        # Preserve the no-raw-history policy while giving the model the neutral
        # action and state-transition contract every round.
        if system_prompt is None:
            system_prompt = agentmemory_action_system_prompt()
        if not isinstance(system_prompt, str) or not system_prompt.strip():
            raise ValueError("Latest-observation system prompt must not be empty.")
        return apply_chat_template(
            tokenizer,
            [
                {"role": "system", "content": system_prompt},
                latest_user_message.to_dict(),
            ],
        )
    
    
    def add_assistant_message(
        self,
        tokenizer: PreTrainedTokenizer,
        content: str,
        format: Literal["qwen"] = "qwen",
    ) -> None:
        msg = Message(role='assistant', content=content)
        self.messages.append(msg)
        assert format in self.format_config.keys(), f"format {format} not supported"
        prefix_msg = self.format_config[format]["assistat_prefix_msg"]
        prefix_token_ids = tokenizer.encode(prefix_msg, add_special_tokens=False)
        suffix_msg = self.format_config[format]["assistat_suffix_msg"]
        suffix_token_ids = tokenizer.encode(suffix_msg, add_special_tokens=False)
        response = tokenizer.encode(content, add_special_tokens=False)
        if self.input_ids[-len(prefix_token_ids) :] == prefix_token_ids:
            append_token_ids = response
            _loss_mask = [1] * len(response)
        elif self.input_ids[-len(suffix_token_ids) :] == suffix_token_ids:
            append_token_ids = prefix_token_ids + response
            _loss_mask = [0] * len(prefix_token_ids) + [1] * len(response)
        else:
            max_len = max(len(prefix_token_ids), len(suffix_token_ids))
            raise ValueError(
                f"""Unsupported end of message format:
                {tokenizer.decode(self.input_ids[-max_len:])}, {tokenizer.decode(self.input_ids)=}"""
            )
        append_token_ids += suffix_token_ids
        _loss_mask += [1] * len(suffix_token_ids)
        self.input_ids += append_token_ids
        _attention_mask = [1] * len(append_token_ids)
        self.attention_mask += _attention_mask
        _delta_position_ids = [pos_id for pos_id in range(1, len(append_token_ids) + 1)]
        last_position_ids = self.position_ids[-1]
        _position_ids = [pos_id + last_position_ids for pos_id in _delta_position_ids]
        self.loss_mask += _loss_mask
        self.position_ids += _position_ids
        assert len(self.input_ids) == len(self.attention_mask) == len(self.position_ids) == len(self.loss_mask), f"""Rollout Handler has different length of {len(self.input_ids)=}, 
            {len(self.attention_mask)=}, {len(self.position_ids)=}, {len(self.loss_mask)=}"""
        
    def add_user_message(
        self,
        tokenizer: PreTrainedTokenizer,
        content: str,
        format: Literal["qwen"] = "qwen",
    ) -> None:
        msg = Message(role='user', content=content)
        self.messages.append(msg)
        assert format in self.format_config.keys(), f"format {format} not supported"
        prefix_msg = self.format_config[format]["user_prefix_msg"]
        prefix_token_ids = tokenizer.encode(prefix_msg, add_special_tokens=False)
        suffix_msg = self.format_config[format]["user_suffix_msg"]
        suffix_token_ids = tokenizer.encode(suffix_msg, add_special_tokens=False)
        content_token_ids = tokenizer.encode(content, add_special_tokens=False)

        if self.input_ids[-len(prefix_token_ids) :] == prefix_token_ids:
            append_token_ids = content_token_ids
            _loss_mask = [0] * len(content_token_ids)
        elif self.input_ids[-len(suffix_token_ids) :] == suffix_token_ids:
            append_token_ids = prefix_token_ids + content_token_ids
            _loss_mask = [0] * len(prefix_token_ids) + [0] * len(content_token_ids)
        else:
            max_len = max(len(prefix_token_ids), len(suffix_token_ids))
            raise ValueError(
                f"""Unsupported end of message format:
                {tokenizer.decode(self.input_ids[-max_len:])}, {tokenizer.decode(self.input_ids)=}"""
            )

        append_token_ids += suffix_token_ids
        _loss_mask += [0] * len(suffix_token_ids)
        self.input_ids += append_token_ids
        _attention_mask = [1] * len(append_token_ids)
        self.attention_mask += _attention_mask
        _delta_position_ids = [pos_id for pos_id in range(1, len(append_token_ids) + 1)]
        last_position_ids = self.position_ids[-1]
        _position_ids = [pos_id + last_position_ids for pos_id in _delta_position_ids]
        self.loss_mask += _loss_mask
        self.position_ids += _position_ids
        assert len(self.input_ids) == len(self.attention_mask) == len(self.position_ids) == len(self.loss_mask), f"""Rollout Handler has different length of {len(self.input_ids)=},
            {len(self.attention_mask)=}, {len(self.position_ids)=}, {len(self.loss_mask)=}"""
        
    def truncate_output_ids(self) -> None:
        self.input_ids = self.input_ids[: self.max_model_len]
        self.attention_mask = self.attention_mask[: self.max_model_len]
        self.position_ids = self.position_ids[: self.max_model_len]
        self.loss_mask = self.loss_mask[: self.max_model_len]
        self.response_ids = self.input_ids[len(self.prompt_ids) :][: self.max_response_len]
        self.response_attention_mask = self.attention_mask[len(self.prompt_attention_mask) :][: self.max_response_len]
        self.response_position_ids = self.position_ids[len(self.prompt_position_ids) :][: self.max_response_len]
        self.response_loss_mask = self.loss_mask[len(self.prompt_loss_mask) :][: self.max_response_len]
