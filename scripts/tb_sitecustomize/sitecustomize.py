import inspect

from google.protobuf import json_format


_original_message_to_json = json_format.MessageToJson
_signature = inspect.signature(_original_message_to_json)

if "including_default_value_fields" not in _signature.parameters:

    def _message_to_json_compat(message, *args, **kwargs):
        include_defaults = kwargs.pop("including_default_value_fields", None)
        if include_defaults is not None:
            kwargs.setdefault(
                "always_print_fields_with_no_presence",
                include_defaults,
            )
        return _original_message_to_json(message, *args, **kwargs)

    json_format.MessageToJson = _message_to_json_compat
