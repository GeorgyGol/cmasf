"""Extends pandas DataFrame class for UPSERT (or INSERT OR UPDATE) data to some database

    This file can also be imported as a module and contains the following functions:

    * iterate_group - useful function for iterating throw list by group with 2 or more elements

    ...and class:
    * DataFrameDATA - child of pandas.DataFrame with re-defned 'to_sql' function

"""

__author__ = "G. Golyshev"
__copyright__ = "CMASF 2020"
__version__ = "1.0.0"
__maintainer__ = "G. Golyshev"
__email__ = "g.golyshev@forecast.ru"
__status__ = "Production"


import pandas as pd
import sqlalchemy as sa
import cmasf.serv as srv


# def iterate_group(iterator, count):
#     itr = iter(iterator)
#     for i in range(0, len(iterator), count):
#         yield iterator[i:i + count]

class DataFrameDATA(pd.DataFrame):

    not_country=[]
    name=''


    @property
    def _constructor(self):
        return DataFrameDATA

    def to_sql(self, name, con, flavor='sqlite', schema=None, if_exists='fail', index=True,
               index_label=None, chunksize=100, dtype=None):
        """Re-defined original function for UPSERT operations
            all difference in params:
                if_exists: it can be
                    - 'replace' for delete table in database if it ezists, create new and fiil it by data from self dataframe
                                indexes of dataframe will be indexes of the new table.
                    -  'append', 'ignore' for append not existing in table data and ignoring existing
                    -  'update', 'upsert' for insert not existing data and update existing. find existing data by
                                self dataframe indexses
                chunksize: writing to database by packages of 'chunksize' records
        """
        def drop_table(strTName):
            meta = sa.MetaData(bind=con)
            try:
                tbl_ = sa.Table(strTName, meta, autoload=True, autoload_with=con)
                tbl_.drop(con, checkfirst=False)
            except:
                pass

        def get_data_dict(strDateFormat='%Y-%m-%d'):
            lst_date = self.reset_index().select_dtypes(include='datetime')
            vals = self.reset_index().to_dict(orient='records')
            for v in vals:
                for d in lst_date:
                    v[d] = v[d].strftime('%Y-%m-%d')
            return vals

        def create_table(strTName, lstIndNames):

            def type_to_sqlA(lstName, sqType):
                l = len(lstName)
                return dict(zip(lstName, [sqType] * l))

            dctReplace={'int':sa.Integer, 'int64':sa.Integer, 'datetime64[ns]':sa.String,
                        'datetime':sa.String, 'float':sa.Float, 'object':sa.String}

            dct_trans = type_to_sqlA(self.select_dtypes(include='int').columns.tolist(), sa.Integer)
            dct_trans.update(type_to_sqlA(self.select_dtypes(include='int64').columns.tolist(), sa.Integer))
            dct_trans.update(type_to_sqlA(self.select_dtypes(include='datetime').columns.tolist(), sa.String))
            dct_trans.update(type_to_sqlA(self.select_dtypes(include='float').columns.tolist(), sa.Float))
            dct_trans.update(type_to_sqlA(self.select_dtypes(include='object').columns.tolist(), sa.String))

            try:
                dct_trans_indexes = {n: dctReplace[str(self.index.get_level_values(n).dtype)] for n in self.index.names}
            except AttributeError:
                dct_trans_indexes = {self.index.name: dctReplace[str(self.index.dtype)]}

            lstIndexes=[sa.Column(k, v, primary_key=True, nullable=False, autoincrement=False) for k, v in dct_trans_indexes.items()]
            lstDBCols=[sa.Column(k, v) for k, v in dct_trans.items()]

            columns=lstIndexes+lstDBCols

            metadata = sa.MetaData(bind=con)

            bname_t = sa.Table(strTName, metadata, *columns)
            metadata.create_all()

        def buff_insert(alch_table, insert_prefix, values, buff_size=chunksize):
            for i in srv.iterate_group(values, buff_size):
                inserter = alch_table.insert(prefixes=insert_prefix, values=i)
                con.execute(inserter)

        if if_exists == 'replace':
            drop_table(name)
            if_exists = 'fail'

        if not con.dialect.has_table(con, name):
            create_table(name, self.index.names)

        meta = sa.MetaData(bind=con)
        tbl_names = sa.Table(name, meta, autoload=True, autoload_with=con)

        vals = get_data_dict()

        inserter = None

        if flavor == 'mysql':
            if if_exists in ['append', 'ignore']:
                inserter = tbl_names.insert(prefixes=['IGNORE'], values=vals)
            elif if_exists in ['update', 'upsert']:
                ins_state = sa.dialects.mysql.insert(tbl_names).values(vals)
                inserter = ins_state.on_duplicate_key_update(Date=ins_state.inserted.Date)
            elif if_exists == 'fail':
                inserter = tbl_names.insert(values=vals)
            con.execute(inserter)

        if flavor == 'sqlite':
            if if_exists in ['append', 'ignore']:
                # inserter = tbl_names.insert(prefixes=['OR IGNORE'], values=vals)
                buff_insert(tbl_names, ['OR IGNORE'], vals, buff_size=chunksize)
            elif if_exists in ['update', 'upsert']:
                buff_insert(tbl_names, ['OR REPLACE'], vals, buff_size=chunksize)
                # inserter = tbl_names.insert(prefixes=['OR REPLACE'], values=vals)
            elif if_exists == 'fail':
                buff_insert(tbl_names, None, vals, buff_size=chunksize)