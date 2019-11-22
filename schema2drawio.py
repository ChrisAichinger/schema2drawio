#!/usr/bin/python3

import sys
import argparse
import collections
from fnmatch import fnmatch
import jinja2
import numpy as np
import sqlalchemy

QUERY_TABLES = dict(
    postgresql="""
        SELECT c.table_schema, c.table_name, v.table_name IS NOT NULL AS is_view, c.column_name, c.udt_name
          FROM information_schema.columns c
     LEFT JOIN information_schema.views v ON c.table_schema = v.table_schema AND c.table_name = v.table_name
         WHERE c.table_schema NOT IN ('information_schema', 'pg_catalog')
      ORDER BY c.table_schema, c.table_name, c.ordinal_position;
    """,
    mysql="""
         SELECT c.table_schema, c.table_name, c.column_name, c.data_type, v.table_name IS NOT NULL AS is_view
           FROM information_schema.columns c
      LEFT JOIN information_schema.views v ON c.table_schema = v.table_schema AND c.table_name = v.table_name
          WHERE c.table_schema NOT IN ('information_schema', 'pg_catalog')
       ORDER BY c.table_schema, c.table_name, c.ordinal_position;
    """,
)

QUERY_FKS = dict(
    postgresql="""
        SELECT DISTINCT
             tc.table_schema,
             tc.table_name,
             kcu.column_name,
             ccu.table_schema AS foreign_table_schema,
             ccu.table_name AS foreign_table_name,
             ccu.column_name AS foreign_column_name
        FROM information_schema.table_constraints AS tc
        JOIN information_schema.key_column_usage AS kcu
              ON tc.constraint_name = kcu.constraint_name
             AND tc.table_schema = kcu.table_schema
             AND tc.table_name = kcu.table_name
        JOIN information_schema.constraint_column_usage AS ccu
              ON ccu.constraint_name = tc.constraint_name
             AND ccu.table_schema = tc.table_schema
       WHERE tc.constraint_type = 'FOREIGN KEY';
    """,
    mysql="""
       SELECT table_schema, table_name, column_name, referenced_table_schema, referenced_table_name, referenced_column_name
       FROM   information_schema.key_column_usage
       WHERE  referenced_column_name is not null;
    """,
)

TEMPLATE = jinja2.Template("""
<?xml version="1.0" encoding="UTF-8"?>
{% set colheight = 26 %}
{% set colwidth = 200 %}
{% set max_table_height = colheight * (1 + max_columns) %}
{% set padding_x = max_table_height - colwidth %}
{% set padding_y = 10 %}
{% set table_style =  'swimlane;childLayout=stackLayout;horizontal=1;horizontalStack=0;fillColor=#dae8fc;swimlaneFillColor=#ffffff;strokeColor=#6c8ebf;align=center;fontSize=14;rounded=1;startSize={};'.format(colheight) %}
{% set view_style =   'swimlane;childLayout=stackLayout;horizontal=1;horizontalStack=0;fillColor=#ffe6cc;swimlaneFillColor=#ffffff;strokeColor=#d79b00;align=center;fontSize=14;rounded=1;startSize={};'.format(colheight) %}
{% set column_style = 'text;strokeColor=none;fillColor=none;spacingLeft=4;spacingRight=4;overflow=hidden;rotatable=0;points=[[0,0.5],[1,0.5]];portConstraint=eastwest;fontSize=12;' %}
{% set fk_style =     'edgeStyle=entityRelationEdgeStyle;fontSize=12;html=1;endArrow=none;startArrow=none;' %}

<mxfile host="www.draw.io" version="12.2.0" type="device" pages="1">
  <diagram id="W1jUUYN19i4ggJYnhAQF" name="Page-1">
    <mxGraphModel dx="863" dy="531" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="1169" pageHeight="827" math="0" shadow="0">
      <root>
        <mxCell id="0"/>
        <mxCell id="1" parent="0"/>

        {% for table in tables %}
            <mxCell
              id="{{table.qualified_name}}"
              value="{{table.display_name}}"
              style="{{view_style if table.is_view else table_style}}"
              vertex="1"
              parent="1"
            >
              <mxGeometry
                x="{{ (positions[table.qualified_name][0] if positions else loop.index0) * (colwidth + padding_x) }}"
                y="{{ (positions[table.qualified_name][1] if positions else 0) * (max_table_height + padding_y) }}"
                width="{{colwidth}}"
                height="{{colheight * (1 + len(table.columns))}}"
                as="geometry"/>
            </mxCell>
            {% for col in table.columns %}
                <mxCell
                  id="{{col.qualified_name}}"
                  value="{{col.name}} ({{col.dtype}})"
                  style="{{column_style}}"
                  vertex="1"
                  parent="{{table.qualified_name}}"
                >
                  <mxGeometry
                    y="{{colheight * loop.index}}"
                    width="{{colwidth}}"
                    height="{{colheight}}"
                    as="geometry"/>
                </mxCell>
            {% endfor %}
        {% endfor %}

        {% for fk in fks %}
            <mxCell
              id="{{"fk.{}-{}".format(fk.src_column_qualified_name, fk.dst_column_qualified_name)}}"
              value=""
              style="{{fk_style}}"
              edge="1"
              parent="1"
              source="{{fk.src_column_qualified_name}}"
              target="{{fk.dst_column_qualified_name}}"
            >
              <mxGeometry width="100" height="100" relative="1" as="geometry"></mxGeometry>
            </mxCell>
        {% endfor %}

      </root>
    </mxGraphModel>
  </diagram>
</mxfile>
""")


class Table:
    def __init__(self, schema, name, is_view=False):
        self.schema = schema
        self.name = name
        self.is_view = is_view
        self.columns = []
        self.qualified_name = "{}.{}".format(self.schema, self.name)
        self.display_name = self.name if self.schema == 'public' else self.qualified_name

    def __repr__(self):
        return "<{} {}>".format(self.__class__.__name__, self.qualified_name)


class Column:
    def __init__(self, table, name, dtype):
        self.name = name
        self.dtype = dtype
        self.qualified_name = "{}.{}".format(table.qualified_name, name)

    def __repr__(self):
        return "<{} {}>".format(self.__class__.__name__, self.qualified_name)


class ForeignKey:
    def __init__(self, src_schema, src_table, src_column, dst_schema, dst_table, dst_column):
        self.src_schema = src_schema
        self.src_table = src_table
        self.src_column = src_column
        self.dst_schema = dst_schema
        self.dst_table = dst_table
        self.dst_column = dst_column

        src_table = Table(src_schema, src_table)
        dst_table = Table(dst_schema, dst_table)
        self.src_table_qualified_name = src_table.qualified_name
        self.dst_table_qualified_name = dst_table.qualified_name
        self.src_column_qualified_name = Column(src_table, src_column, '').qualified_name
        self.dst_column_qualified_name = Column(dst_table, dst_column, '').qualified_name

    def __repr__(self):
        return "<{} {} - {}>".format(self.__class__.__name__,
                                     self.src_column_qualified_name,
                                     self.dst_column_qualified_name)


def load_tables(con):
    tables = {}
    r = con.execute(QUERY_TABLES[con.dialect.name])
    for row in r:
        t = Table(row[0], row[1], row[2])
        t = tables.setdefault(t.qualified_name, t)
        t.columns.append(Column(t, row[3], row[4]))
    return tables


def load_foreign_keys(con, tables):
    r = con.execute(QUERY_FKS[con.dialect.name])
    return [ForeignKey(*row) for row in r.fetchall()]


def apply_filter(tables, fks, schema, exclude_schema, table_filter, exclude_table_filter):
    def should_accept(t):
        accept = True
        if schema and not any(fnmatch(t.schema, s) for s in schema):
            accept = False
        if exclude_schema and any(fnmatch(t.schema, s) for s in exclude_schema):
            accept = False
        if table_filter and not any(fnmatch(t.name, tn) for tn in table_filter):
            accept = False
        if exclude_table_filter and any(fnmatch(t.name, tn) for tn in exclude_table_filter):
            accept = False
        return accept

    tables = {k: t for k, t in tables.items() if should_accept(t)}
    fks = [fk for fk in fks
           if fk.src_table_qualified_name in tables and fk.dst_table_qualified_name in tables]
    return tables, fks


class OptRectGeneticOneRandomPermutation:
    """Optimize placement of nodes on a rectangular grid: genetic algorithm, single random permutation/gen"""
    def __init__(self, nodes, edges, width=None):
        self.nodes = list(nodes)
        self.edges = list(edges)
        self.adjacency_list = np.array([(self.nodes.index(a), self.nodes.index(b)) for a, b in self.edges])
        self.width = width
        if self.width is None:
            self.width = int(np.ceil(np.sqrt(len(self.nodes) * 2)))  # Aim for ~50% fill rate
        self.state_len = self.width * self.width
        distances_shape = (self.state_len, self.state_len)
        self.distances = np.fromfunction(self._calc_distances, distances_shape)

    def calc_cost(self, state):
        allocated_edges = np.apply_along_axis(state.__getitem__, 0, self.adjacency_list)
        allocated_distances = self.distances[tuple(allocated_edges.T)]
        return np.linalg.norm(allocated_distances)

    def initialize_state(self):
        return np.random.permutation(np.arange(self.state_len))

    def update_state(self, state, generation, num_generations):
        selector = np.random.permutation(np.arange(self.state_len))[:2]
        state[selector] = np.flip(state[selector])
        return state

    def keep_state(self, cost, last_cost, state, last_state):
        return cost < last_cost

    def get_node_positions(self, state):
        return {k: self._calc_coords(state[i]) for i, k in enumerate(self.nodes)}

    def plot(self, state, **kwargs):
        import networkx as nx
        return nx.draw_networkx(self.get_networkx_graph(), pos=self.get_positions(state), **kwargs)

    def get_networkx_graph(self):
        import networkx as nx
        g = nx.Graph()
        for n in self.nodes:
            g.add_node(n)
        for e1, e2 in self.edges:
            g.add_edge(e1, e2)
        return g

    def _calc_coords(self, idx):
        return idx % self.width, idx // self.width

    def _calc_distances(self, i, j):
        ix, iy = self._calc_coords(i)
        jx, jy = self._calc_coords(j)
        return np.linalg.norm([ix - jx, iy - jy], axis=0)


def optimize_one_population(o, num_generations):
    best_state = o.initialize_state()
    best_cost = o.calc_cost(best_state)
    states = [(best_cost, best_state)]
    for i in range(num_generations):
        last_cost, last_state = states[-1]
        state = o.update_state(last_state.copy(), i, num_generations)
        cost = o.calc_cost(state)
        if cost < best_cost:
            best_cost = cost
            best_state = state
        if o.keep_state(cost, last_cost, state, last_state):
            states.append((cost, state))
        else:
            states.append((last_cost, last_state))
        states.append((best_cost, best_state))

    return best_cost, best_state, states


def optimize(o, num_populations=10, num_generations=4000):
    cost_series = []
    optimized_results = []
    for j in range(num_populations):
        best_cost, best_state, states = optimize_one_population(o, num_generations=num_generations)
        optimized_results.append((best_cost, best_state))

    best_cost, best_state = sorted(optimized_results, key=lambda tup: tup[0])[0]
    return best_cost, best_state


def optimize_table_positions(tables, fks):
    optimizer_fks = [(fk.src_table_qualified_name, fk.dst_table_qualified_name) for fk in fks]
    optimizer = OptRectGeneticOneRandomPermutation(tables.keys(), optimizer_fks)
    best_cost, best_state = optimize(optimizer)
    return optimizer.get_node_positions(best_state)


def main():
    parser = argparse.ArgumentParser(description='Create Draw.IO ER schema for a database')
    parser.add_argument('dburl', metavar='DBURL',
                        help='SQLAlchemy connection string (e.g. postgresql://user:pass@host:port/dbname)')
    parser.add_argument('ofile', metavar='OUTPUT', type=argparse.FileType('w'),
                        help='Output XML file name')
    parser.add_argument('-n', '--schema', action='append',
                        help='schema to include (can be specified multiple times, cf. pg_dump man page)')
    parser.add_argument('-N', '--exclude-schema', action='append',
                        help='schema to exclude (can be specified multiple times, cf. pg_dump man page)')
    parser.add_argument('-t', '--table', action='append',
                        help='table to include (can be specified multiple times, cf. pg_dump man page)')
    parser.add_argument('-T', '--exclude-table', action='append',
                        help='table to exclude (can be specified multiple times, cf. pg_dump man page)')
    args = parser.parse_args()

    con = sqlalchemy.create_engine(args.dburl)
    if con.dialect.name not in QUERY_TABLES:
        msg = "Database dialect not supported: %s; available: %r\n"
        msg += "You can add support for new dialects easily by adding two SQL queries to the top of %r."
        msg = msg.format(con.dialect.name, ', '.join(QUERY_TABLES.keys()), __file__)
        print(msg, file=sys.stderr)

    tables = load_tables(con)
    fks = load_foreign_keys(con, tables)
    tables, fks = apply_filter(tables, fks, args.schema, args.exclude_schema, args.table, args.exclude_table)
    for t in tables.values():
        print(t.qualified_name, file=sys.stderr)

    positions = {}
    if tables:
        positions = optimize_table_positions(tables, fks)

    xml = TEMPLATE.render(tables=tables.values(), fks=fks, len=len, positions=positions, max_columns=max(len(t.columns) for t in tables.values()))
    args.ofile.write(xml)

if __name__ == '__main__':
    main()
