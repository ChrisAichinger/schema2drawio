=======
schema2drawio - Generate SQL schema diagrams in Draw.IO format
=======

schema2drawio creates a Draw.IO diagram describing an SQL database schema. The database metadata
is extracted automatically from the database server.

Example usage::

    ./schema2drawio.py 'postgresql://user:passwd@host:port/dbname' output_fname.drawio
    ./schema2drawio.py 'mysql+mysqlconnector://user:passwd@host:port/dbname' output_fname.drawio

Currently, Mysql and Postgres are supported. Adding support for further databases is easy.

Release notes
-------------

* **0.1.0**, released on 2019-11-22

  - Initial release
