import pandas as pd
import sqlite3 as lite
#engine = sqla.create_engine('sqlite:///football.sqlite')
conn = lite.connect('football.sqlite')

#matchData  = pd.read_sql('Match', conn)

#meta = sqla.MetaData()
cur = conn.cursor()
cur.execute('SELECT * FROM Match')

T_Match = cur.fetchall()

df = pd.DataFrame(T_Match)
#for row in T_Match[0]:
#	print row
EPL = df[df[2].isin([1729])]
cols = list(EPL.columns.values)

EPL.to_csv('eplmatches.csv')