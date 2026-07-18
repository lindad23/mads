"""Launch TensorBoard with a protobuf 5 compatibility shim."""

from google.protobuf import json_format


_message_to_json = json_format.MessageToJson


def _message_to_json_compat(message, *args, **kwargs):
    if "including_default_value_fields" in kwargs:
        value = kwargs.pop("including_default_value_fields")
        kwargs.setdefault("always_print_fields_with_no_presence", value)
    return _message_to_json(message, *args, **kwargs)


json_format.MessageToJson = _message_to_json_compat

from tensorboard.main import run_main


if __name__ == "__main__":
    run_main()
