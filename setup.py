from setuptools import setup

package_name = 'fura2a'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your_email@example.com',
    description='A ROS 2 package with both Python and C++ nodes',
    license='Your License',
    entry_points={
        'console_scripts': [
            'simple_publisher = fura2a.simple_publisher:main',
            'lqr = fura2a.lqr:main',
        ],
    },
)
