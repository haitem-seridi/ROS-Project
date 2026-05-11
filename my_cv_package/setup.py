from setuptools import find_packages, setup

package_name = 'my_cv_package'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='turtle',
    maintainer_email='turtle@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
    'console_scripts': [
        'cv_plot = my_cv_package.cv_plot:main',
        'line_follower_node = my_cv_package.line_follower_node:main',
        'hsv_tuner_node     = my_cv_package.hsv_tuner_node:main',
    ],

    },
)
