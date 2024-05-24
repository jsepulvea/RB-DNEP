=============================
RB-DNEP: Robust Bilevel DNEP
=============================

Overview
========
``RB-DNEP`` is a Python package designed to implement a robust bilevel optimization model for distribution network expansion planning. The package addresses the interdependent planning needs of active distribution networks (ADNs) and local energy communities (LECs), incorporating uncertainty in demand and solar irradiance fluctuations.

The model is structured into two levels: the upper-level problem, handled by the distribution system operator, involves investment decisions in infrastructure. The lower-level, tackled by local energy communities, focuses on the deployment of distributed energy resources, considering the constraints imposed by the upper-level decisions.

Installation
============

To install this package, follow these steps:

.. note::
   This guide assumes that you have Python 3 installed on your system.

1. **Clone the repository:**

.. code-block:: bash

    git clone git@github.com:jsepulvea/RB-DNEP.git
    cd RB-DNEP 

2. **Create and activate a virtual environment (recommended):**

.. code-block:: bash

    python3 -m venv venv
    source venv/bin/activate

3. **Install the package in editable mode using pip:**

.. code-block:: bash

    pip install -e .

This will install the package in editable mode, allowing you to make changes to the code and have them immediately reflected without reinstalling the package.

Usage
=====

Documentation
=============

Contributing
============

License
=======

Contact
=======

Acknowledgements
================


Note
====

This project has been set up using PyScaffold 4.5. For details and usage
information on PyScaffold see https://pyscaffold.org/.
