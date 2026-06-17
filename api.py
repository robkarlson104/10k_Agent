
import pandas as pd


sec_cik = "https://www.sec.gov/include/ticker.txt"

cik_df = pd.read_table(sec_cik)

print(cik_df)

