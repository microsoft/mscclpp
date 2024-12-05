.. MSCCL++ documentation master file, created by
   sphinx-quickstart on Tue Sep  5 13:03:46 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to MSCCL++'s documentation!
===================================

MSCCL++ is a GPU-driven communication stack for scalable AI applications. It is designed to provide a high-performance, scalable, and customizable communication stack for distributed GPU applications.

Getting Started
---------------
- Follow the :doc:`quick start <getting-started/quickstart>` for your platform of choice.
- Take a look at the :doc:`tutorials <getting-started/tutorials/index>` to learn how to write your first mscclpp program.

.. toctree::
   :maxdepth: 1
   :caption:  Getting Started
   :hidden:

   getting-started/quickstart
   getting-started/tutorials/index

Design
-------
- :doc:`Design <design/design>` doc for those who want to understand the internals of MSCCL++.
- :doc:`NCCL over MSCCL++ <design/nccl-over-mscclpp>` doc for those who want to understand how to use NCCL over MSCCL++.

.. toctree::
   :maxdepth: 1
   :caption:  Design
   :hidden:

   design/design
   design/nccl-over-mscclpp

Performance
---------------
- We evaluate the performance of MSCCL++ in A100 and H100. Here are some :doc:`performance results <performance/performance-ndmv4>` for all-reduce operations.

.. toctree::
   :maxdepth: 1
   :caption:  Performance
   :hidden:

   performance/performance-ndmv4

C++ API
---------------
- :doc:`mscclpp <api/index>`


.. toctree::
   :maxdepth: 1
   :caption: C++ API
   :hidden:

   api/index

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
