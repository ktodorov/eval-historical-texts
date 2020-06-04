import os
import time
from datetime import datetime

from typing import Callable

from entities.timespan import Timespan

from services.arguments.arguments_service_base import ArgumentsServiceBase
from services.file_service import FileService
from services.data_service import DataService


class CacheService:
    def __init__(
            self,
            arguments_service: ArgumentsServiceBase,
            file_service: FileService,
            data_service: DataService):

        self._arguments_service = arguments_service
        self._file_service = file_service
        self._data_service = data_service

        self._cache_folder = self._file_service.combine_path(
            '.cache',
            self._arguments_service.challenge.value.lower(),
            self._arguments_service.configuration.value.lower(),
            self._arguments_service.language.value.lower(),
            create_if_missing=True)

    def get_item_from_cache(
            self,
            item_key: str,
            callback_function: Callable,
            time_to_keep: Timespan = None) -> object:

        # try to get the cached object
        cached_object = self._data_service.load_python_obj(
            self._cache_folder,
            item_key,
            print_on_error=False,
            print_on_success=False)

        if cached_object is None or self._cache_has_expired(item_key, time_to_keep):
            # if the cached object does not exist or has expired we call
            # the callback function to calculate it and then cache it to the file system
            cached_object = callback_function()
            self.cache_item(cached_object, item_key)

        return cached_object

    def cache_item(self, item: object, item_key: str):
        self._data_service.save_python_obj(
            item,
            self._cache_folder,
            item_key,
            print_success=False)

    def _cache_has_expired(
            self,
            item_key: str,
            time_to_keep: Timespan) -> bool:
        if time_to_keep is None:
            return False

        item_path = os.path.join(self._cache_folder, f'{item_key}.pickle')

        if not os.path.exists(item_path):
            return True

        file_mtime = os.path.getmtime(item_path)
        file_datetime = datetime.fromtimestamp(file_mtime)
        current_datetime = datetime.now()
        datetime_diff = (file_datetime - current_datetime)

        if datetime_diff.microseconds > time_to_keep.milliseconds:
            return True

        return False
