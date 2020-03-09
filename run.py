import sys
import logging

from dependency_injection.ioc_container import IocContainer


if __name__ == '__main__':
    # Configure container:
    container = IocContainer()

    container.logger().addHandler(logging.StreamHandler(sys.stdout))

    # Run application:
    container.main()
