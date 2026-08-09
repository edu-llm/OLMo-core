"""Show what the registry's `olmo` chat template does to our inlined rows, and why we avoid it.

    python3 docs/tool-call/verify/verify_template_choice.py

open-instruct ships a `CHAT_TEMPLATES` registry. Its `olmo` entry emits, for a system message:

    '<|im_start|>system\\n' + content
    {% if message.get('functions', none) is not none %}  ' <functions>' + functions + '</functions>'
    {% else %}  ' You do not currently have access to any functions. <functions></functions>'

Our rows carry **no** `functions` field on purpose -- the schemas are inlined into `content` so the
leakage check can see them. So that `else` branch fires on every row, and appends a flat denial
immediately after the tools we just listed.

The shipped `Olmo-3-7B-Instruct/chat_template.jinja` has no such branch. Two consequences:
our own producer (which reproduces the shipped template byte for byte) is unaffected, and anyone
reaching for open-instruct's converter must not select the registry `olmo` template.
"""

import jinja2

REGISTRY_OLMO_SYSTEM_BRANCH = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}"
    "{{ '<|im_start|>system\n' + message['content'] }}"
    "{% if message.get('functions', none) is not none %}"
    "{{ ' <functions>' + message['functions'] + '</functions><|im_end|>\n' }}"
    "{% else %}"
    "{{ ' You do not currently have access to any functions. <functions></functions><|im_end|>\n' }}"
    "{% endif %}"
    "{% endif %}"
    "{% endfor %}"
)

SCHEMAS = (
    '[{"type":"function","function":{"name":"get_weather","description":"Current conditions."}}]'
)
OUR_ROW = [
    {
        "role": "system",
        "content": (
            "You are a helpful function-calling AI assistant. You are provided with function "
            f"signatures within <functions></functions> XML tags. <functions>{SCHEMAS}</functions>"
        ),
    }
]

env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True)
rendered = env.from_string(REGISTRY_OLMO_SYSTEM_BRANCH).render(messages=OUR_ROW)

print("=== our system message, rendered by the REGISTRY 'olmo' template ===")
print(rendered)
print("=== the damage ===")
denial = "You do not currently have access to any functions."
print(f"  contains our schema block?          {SCHEMAS in rendered}")
print(f"  ALSO contains the denial sentence?  {denial in rendered}")
if SCHEMAS in rendered and denial in rendered:
    print()
    print("  Both. Every row would list the tools and then say there are none.")
    print("  On 40,000 rows that is the whole corpus, silently.")

print()
print("=== why our pipeline is unaffected ===")
print("  Our producer reproduces the SHIPPED chat_template.jinja, which has no such else branch")
print(
    "  (proven byte-identical in verify_render_identity.py). We never call the registry template."
)
print("  This check exists so that nobody reintroduces the hazard by reaching for open-instruct's")
print(
    "  converter with --chat_template_name olmo. If that path is ever needed, pass a name that is"
)
print("  NOT in the registry -- the README's 'olmo123' placeholder -- so it falls back to the")
print("  tokenizer's own template.")
