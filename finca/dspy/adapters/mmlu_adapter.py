from dspy.adapters.base import Adapter

class MMLUAdapter(Adapter):
    def __init__(self):
        super().__init__()

    def format(self, signature, demos, inputs):
        # system_prompt = signature.instructions
        all_fields = signature.model_fields
        all_field_data = [(all_fields[f].json_schema_extra["prefix"], all_fields[f].json_schema_extra["desc"]) for f in all_fields]

        all_field_data_str = "\n".join([f"{p} {d}" for p, d in all_field_data])
        format_instruction_prompt = "\n\n" + "="*20 + f"""\n\nOutput Format:\n\n{all_field_data_str}\n\n""" + "="*20 + "\n\n" 

        all_input_fields = signature.input_fields
        input_fields_data = [(all_input_fields[f].json_schema_extra["prefix"], inputs[f]) for f in all_input_fields]

        input_fields_str = "\n".join([f"{p} {v}" for p, v in input_fields_data])

        print("----------------------------------------------------------------------------------------------------------------")
        print("format_instruction_prompt:")
        print(format_instruction_prompt)
        print("input_fields_str:")
        print(input_fields_str)
        return "system_prompt" + format_instruction_prompt + input_fields_str

    def parse(self, signature, completions, _parse_values=None):
        output_fields = signature.output_fields

        output_dict = {}
        for field in output_fields:
            field_info = output_fields[field]
            prefix = field_info.json_schema_extra["prefix"]

            field_completion = completions.split(prefix.upper())[-1].split("\n")[0].strip(": ")
            output_dict[field] = field_completion

        return output_dict