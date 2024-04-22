import os

PATH = '...' # Change to the path of your folder
os.chdir(PATH)

import sys; print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(['.'])

import json
from ner_config import *


INPUT_PATH = os.path.join(PATH, 'example_input.csv')
OUTPUT_PATH = os.path.join(PATH, 'output')

if not os.path.exists(OUTPUT_PATH):
    os.mkdir(OUTPUT_PATH)


if __name__ == '__main__':

    try:
        text_gen = text_generator_from_csv(INPUT_PATH)

        while 1:
            # Get the text
            row = next(text_gen)
            note_id = row[0]
            text = row[1]

            # Pass the text to the model and generate output
            set_seed(42)

            torch.cuda.empty_cache()

            input_collection = text_to_sections(text)
            output_collection = run_ner(input_collection)
            entity_collection = collect_entities(output_collection)
            text_marked = entity_collection_to_markdown(text, entity_collection)

            # Save results
            with open(os.path.join(OUTPUT_PATH, '%s_output.json' % note_id), 'w') as f:
                json.dump(output_collection, f, indent=4)

            with open(os.path.join(OUTPUT_PATH, '%s_entity.json' % note_id), 'w') as f:
                json.dump(entity_collection, f, indent=4)

            with open(os.path.join(OUTPUT_PATH, '%s_marked.md' % note_id), 'w') as f:
                f.write(text_marked)

    except StopIteration:
        print('All notes processed')
        sys.exit(0)
