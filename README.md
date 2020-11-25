<h1 align="center">Pyronear Risks</h1>
<p align="center">
    <a href="LICENSE" alt="License">
        <img src="https://img.shields.io/badge/License-GPLv3-blue.svg" /></a>
    <a href="https://github.com/pyronear/pyro-vision/actions?query=workflow%3Apython-package">
        <img src="https://github.com/pyronear/pyro-risks/workflows/python-package/badge.svg" /></a>
   <a href="https://www.codacy.com/gh/pyronear/pyro-risks/dashboard?utm_source=github.com&utm_medium=referral&utm_content=pyronear/pyro-risks&utm_campaign=Badge_Grade">
        <img src="https://camo.githubusercontent.com/6361a174bbd36acd5ee8c24b0ef27ba6a84803c2ac9354d57d60d1264d78a31a/68747470733a2f2f6170702e636f646163792e636f6d2f70726f6a6563742f62616467652f47726164652f6532623936393836356539663439633561623934343435643765346132613637" /></a>
    <a href="https://codecov.io/gh/pyronear/pyro-risks">
  		<img src="https://codecov.io/gh/pyronear/pyro-risks/branch/master/graph/badge.svg" /></a>
    <a href="https://github.com/psf/black">
        <img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
    <a href="https://pyronear.github.io/pyro-risks">
  		<img src="https://img.shields.io/badge/docs-available-blue.svg" /></a>
</p>

The pyro-risks project aims at providing the pyronear-platform with a machine learning based wildifire risk forcasting capibility. 

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Getting started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
  - [datasets](#datasets)
- [Examples](#examples)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [Credits](#credits)
- [License](#license)

## Getting started

### Prerequisites

-   Python 3.6 (or more recent)
-   [pip](https://pip.pypa.io/en/stable/)

### Installation

You can install the package from github as follows:

```shell
pip install git+git://github.com/pyronear/pyro-risks
```

## Usage

### datasets

Access all pyro-risks datasets. 

```python
from pyro_risks.datasets import NASAFIRMS, NOAAWeather
firms = NASAFIRMS()
noaa = NOAAWeather()
```

## Examples

You are free to merge the datasets however you want and to implement any zonal statistic you want, but some are already provided for reference. In order to use them check the example scripts options as follows:

```shell
python scripts/example_ERA5_FIRMS.py --help
```

You can then run the script with your own arguments:

```shell
python scripts/example_ERA5_FIRMS.py --type_of_merged departements
```

## Documentation

The full package documentation is available [here](https://pyronear.org/pyro-risks/) for detailed specifications. The documentation was built with [Sphinx](https://www.sphinx-doc.org) using a [theme](https://github.com/readthedocs/sphinx_rtd_theme) provided by [Read the Docs](https://readthedocs.org).

## Contributing

Please refer to the [`CONTRIBUTING`](./CONTRIBUTING.md) guide if you wish to contribute to this project.

## Credits

This project is developed and maintained by the repo owner and volunteers from [Data for Good](https://dataforgood.fr/).

## License

Distributed under the GPLv3 Licenses. See [`LICENSE`](./LICENSE) for more information.