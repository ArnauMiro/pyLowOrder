Welcome to pyLOM's documentation
================================


.. toctree::
   :maxdepth: 1
   :hidden:

   User Guide <user/index>
   API reference <api/modules>
   Building from source <https://github.com/ArnauMiro/pyLowOrder/wiki/Deployment>


**Version**: |version|
   
**Useful links**:
`Source Repository <https://github.com/ArnauMiro/pyLowOrder>`_ |
`Issue Tracker <https://github.com/ArnauMiro/pyLowOrder/issues>`_ |
`Wiki <https://github.com/ArnauMiro/pyLowOrder/wiki>`_ |


**pyLOM** is a Python library for low-order modeling techniques.
This tool is a port of the POD/DMD of the tools from UPM in MATLAB to C/C++ using a python interface. So far POD, DMD and sPOD are fully implemented and work is being done to bring hoDMD and VAEs inside the tool. Please check the wiki for instructions on how to deploy the tool.


.. grid:: 1 1 2 2
    :gutter: 2 3 4 4

    .. grid-item-card::
        :img-top: ../source/_static/index-images/getting_started.svg
        :text-align: center

        Getting started
        ^^^

        See some examples of how to use pyLOM. The examples are short and easy to understand, and they will help you get started with the library.

        +++

        .. button-ref:: auto_examples/index
            :expand:
            :color: secondary
            :click-parent:

            To the examples

    .. grid-item-card::
        :img-top: ../source/_static/index-images/install.svg
        :text-align: center

        Installation guide
        ^^^

        To install the tool you can see the instructions in the wiki. The tool is available in PyPI and can be installed using pip.

        +++

        .. button-link:: https://github.com/ArnauMiro/pyLowOrder/wiki/Deployment 
            :expand:
            :color: secondary
            :click-parent:

            To the installation guide

    .. grid-item-card::
        :img-top: ../source/_static/index-images/api.svg
        :text-align: center

        API reference
        ^^^

        The reference guide contains a detailed description of the functions,
        modules, and objects included in pyLOM. The reference describes how the
        methods work and which parameters can be used. It assumes that you have an
        understanding of the key concepts.

        +++

        .. button-ref:: api/modules
            :expand:
            :color: secondary
            :click-parent:

            To the reference guide

    .. grid-item-card::
        :img-top: ../source/_static/index-images/contributor.svg
        :text-align: center

        Contributor's guide
        ^^^

        Want to add to the codebase? The contributing guidelines will guide you through the
        process of improving pyLOM.

        +++

        .. button-link:: https://github.com/ArnauMiro/pyLowOrder/wiki/Adding-a-new-model
            :expand:
            :color: secondary
            :click-parent:

            To the contributor's guide
