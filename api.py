
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

email = os.getenv("SEC_EMAIL")

sec_cik = "https://www.sec.gov/include/ticker.txt"



cik_df = pd.read_table(sec_cik,)

print(cik_df)


#### your goal should be to just run one ticker and get the output to test.
