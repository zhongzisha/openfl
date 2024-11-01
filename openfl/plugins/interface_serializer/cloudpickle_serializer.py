# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""Cloudpickle serializer plugin."""

import cloudpickle

from openfl.plugins.interface_serializer.serializer_interface import Serializer


class CloudpickleSerializer(Serializer):
    """Serializer API plugin."""

    def __init__(self) -> None:
        """Initialize serializer."""
        super().__init__()

    @staticmethod
    def serialize(object_, filename):
        """
        Serialize an object and save to disk.

        Args:
            object_ (object): The object to be serialized.
            filename (str): The name of the file where the serialized object
                will be saved.

        Returns:
            None
        """
        with open(filename, "wb") as f:
            cloudpickle.dump(object_, f)

    @staticmethod
    def restore_object(filename):
        """
        Load and deserialize an object.

        Args:
            filename (str): The name of the file where the serialized object
                is saved.

        Returns:
            object: The deserialized object.
        """
        with open(filename, "rb") as f:
            return cloudpickle.load(f)
