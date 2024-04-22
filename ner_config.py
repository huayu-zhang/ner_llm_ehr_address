"""
Design of data structure/data flow

Input - str

Encoded template - tensor

Single Output - str

Input collection
[{<section_name>: str, <section_start>: int, <section_end>: int, <section_text>: str}]

Output collection
[{<section_name>: str, <section_start>: int, <section_end>: int,
<section_text>: str, <generated_text>: str, <generated_parsed>: dict}, ...]

Generated parsed
{'Phenotypes': [{<quote>: <description>}, ...]

Entity collection

[{'type': 'entity', 'section_name': <section_name>, 'string': <quote>,
'description': <description>, 'start': int, 'end': int}]

Brat Standoff
f'''T{enumerate}\t{category} {start} {end}\t{entity_name}\n...'''



List of functions

text_to_sections(
text, p_section
) -> Input collection


run_ner(
input_collection, model, tokenizer, template
) -> Output collection

run(

parse(
<generated_text>
) -> <generated_parsed>

collect_entities(
Output collection
) -> Entity collection

entity_collection_to_brat_standoff(
Entity collection
) -> Brat standoff

[OPTIONAL]

entity_collections_to_markdown(

) -> Text for visualization


To-do
For memory efficiency, when input is too long, sub_sections should be created to fit input to memory

"""
import sys

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
import re
import csv

import yaml

from copy import deepcopy


P_SECTION = None

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
MODEL = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL.to(DEVICE)

TEMPLATE = [
    {"role": 'user',
     'content':
         (
             "I would like you to quote ALL addresses from a text.\n"
             "If there are multiple addresses, please quote each of them. \n"
             "The output should be given in .yaml format.\n"
         )
     },
    {"role": "assistant",
     "content": "Sure! Give me an example and I will try to help. \n"},
    {"role": "user",
     "content":
         (
             "Example: \n"

             "Text: \n"
             "This is a lovely 56 yo lady with chronic conditions including "
             "T2DM and heart attack. But no complaints of depression."
             "Her daughter lives at 21 Dalkeith Rd, Edinburgh EH16 5BB\n"
             "The patient herself recently moved to 100 Telford Rd, Edinburgh EH4 2NF\n\n"

             "Example output: \n"
             '```yaml\n'
             '     - "21 Dalkeith Rd, Edinburgh EH16 5BB"\n'
             '     - "100 Telford Rd, Edinburgh EH4 2NF"\n'
             "```\n"
         )
     },
    {"role": "assistant",
     "content": "Now give me the text for my task in triple quote"
     },
    {
        "role": "user",
        "content": (
            "Your task: \n"
            '"""\n'
            "[****PROMPT****]\n"
            '"""'
        )
    }
]


def text_to_sections(text, p_section=P_SECTION):
    """
    Generate text split by pattern in `p_section`

    :param text: str
    :param p_section: re pattern
    :return: generator of split text
    """

    if p_section is None:
        input_collection = [
            {
                'section_name': 'full_text',
                'section_start': 0,
                'section_end': len(text),
                'section_text': text
            }
        ]

    else:
        ms = list(re.finditer(p_section, text))

        starts = [m.start() for m in ms]
        ends = starts[1:] + [len(text)]

        input_collection = [
            {
                'section_name': m.group(1),
                'section_start': start,
                'section_end': end,
                'section_text': text[start:end]
            }
            for m, start, end in zip(ms, starts, ends)
        ]

    return input_collection


def make_prompt(text):
    template = deepcopy(TEMPLATE)

    template[-1]['content'] = template[-1]['content'].replace('[****PROMPT****]', text)

    encoded = TOKENIZER.apply_chat_template(
        template, add_generation_prompt=True, return_tensors="pt").to(DEVICE)

    return encoded


def text_generator_from_csv(csv_path):

    with open(csv_path, 'r') as f:
        csv_reader = csv.reader(f)

        headers = next(csv_reader)

        print(headers)

        while 1:
            try:
                yield next(csv_reader)
            except StopIteration:
                return


def output_generate(text):

    set_seed(42)

    torch.cuda.empty_cache()

    encoded = make_prompt(text)

    outputs = MODEL.generate(
        encoded,
        max_new_tokens=500,
        pad_token_id=TOKENIZER.eos_token_id,
        num_beams=20,
        do_sample=True,
        temperature=0.5
    )

    output_str = TOKENIZER.decode(outputs[0])

    del encoded
    del outputs

    p_conv = re.compile('\[.?INST]')

    start_of_gen = list(re.finditer(p_conv, output_str))[-1].end()

    generated_text = output_str[start_of_gen:]

    return generated_text


def safe_parsing_output(text):

    try:
        parsed_yaml = re.search(r'```yaml\n((.*\n)+.*)```', text).group(1)
        return yaml.safe_load(parsed_yaml)

    except AttributeError:
        try:
            text = text.replace('*', '    -')
            return yaml.safe_load(text)
        except:
            print(text)
            return 'Cannot parse'
    except:
        print(text)
        return 'Cannot parse'


def run_ner(input_collection):

    for collect in input_collection:

        print("Input length: %s" % len(collect['section_text']))

        collect['generated_text'] = output_generate(collect['section_text'][:2500])

        collect['generated_parsed'] = safe_parsing_output(collect['generated_text'])

    output_collection = input_collection

    return output_collection


def collect_entities(output_collection):

    records = []

    for collect in output_collection:

        if isinstance(collect['generated_parsed'], list):

            for quote in collect['generated_parsed']:
                ms = list(re.finditer(pattern=quote, string=collect['section_text']))

                for m in ms:

                    record = {
                        'type': 'entity',
                        'section_name': collect['section_name'],
                        'string': quote,
                        'description': 'AddressInfo',
                        'start': collect['section_start'] + m.start(),
                        'end': collect['section_start'] + m.end()
                    }
                    records.append(record)

        else:
            print('The generated results could not be parsed')

    entity_collection = sorted(records, key=lambda d: d['start'])

    return entity_collection


def entity_collection_to_brat_standoff(entity_collection):

    standoff_rows = []

    for i, entity in enumerate(entity_collection):

        row = f'''T{i+1}\t{'Phenotype'} {entity['start']} {entity['end']}\t{entity['description'][:40]}'''

        standoff_rows.append(row)

    standoff_file = '\n'.join(standoff_rows)

    return standoff_file


def format_entity_string(entity, html_style=('<mark>', '</mark>')):

    return f'''{html_style[0]}{entity['string']}<sup>{entity['description'][:40]}</sup>{html_style[1]}'''


def entity_collection_to_markdown(text, entity_collection):

    text_segments = []

    segment_start = 0

    for entity in entity_collection:
        text_segments.append(text[segment_start:entity['start']])

        text_segments.append(
            format_entity_string(entity)
        )

        segment_start = entity['end']

    text_segments.append(text[segment_start:])

    text_marked = ''.join(text_segments).replace('\n', '<br>').replace('_', '=')

    return text_marked

