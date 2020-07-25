import os
import time
from datetime import datetime

from typing import Callable
import urllib.request

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
            callback_function: Callable = None,
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
            if callback_function is None:
                return None

            cached_object = callback_function()
            self.cache_item(item_key, cached_object)

        return cached_object

    def load_file_from_cache(
            self,
            item_key: str) -> object:
        filepath = os.path.join(self._cache_folder, item_key)
        with open(filepath, 'rb') as cached_file:
            result = cached_file.read()
            return result

    def cache_item(self, item_key: str, item: object, overwrite: bool = True):
        if not overwrite and self.item_exists(item_key):
            return

        self._data_service.save_python_obj(
            item,
            self._cache_folder,
            item_key,
            print_success=False)

    def item_exists(self, item_key: str) -> bool:
        result = self._data_service.check_python_object(self._cache_folder, item_key)
        return result

    def download_and_cache(self, item_key: str, download_url: str, overwrite: bool = True) -> bool:
        if not overwrite and self.item_exists(item_key):
            return True

        try:
            download_file_path = os.path.join(self._cache_folder, item_key)
            urllib.request.urlretrieve(
                download_url,
                download_file_path)
        except:
            print(
                f'There was error downloading file from url \'{download_url}\'')

            return False

        return True

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
