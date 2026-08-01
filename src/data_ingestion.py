import pandas as pd
import zipfile
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class LocalMFPipeline:
    """
    Handles the high-speed extraction and loading of mutual fund data 
    directly from a compressed ZIP archive into pandas DataFrames.
    """
    def __init__(self, zip_path):
        self.zip_path = zip_path

    def load_user_funds(self, user_fund_names):
        logging.info(f"Opening local Point-in-Time database: {self.zip_path}")
        fund_data_dict = {}
        
        # Normalize user inputs
        user_lowers = [name.lower().strip() for name in user_fund_names]

        # Context Managers (with statement) ensure file locks are safely released from RAM
        with zipfile.ZipFile(self.zip_path, 'r') as z:
            # 1. Map the directory of the ZIP
            all_files = [f for f in z.namelist() if f.endswith('.csv')]
            logging.info(f"Found {len(all_files)} total files in the archive.")

            # 2. Match the User Strings against the Filenames
            matched_files = []
            for file_name in all_files:
                clean_file_name = file_name.lower().replace("_", " ")
                for u_name in user_lowers:
                    if u_name in clean_file_name:
                        matched_files.append(file_name)
                        break
            matched_files = list(set(matched_files))
            logging.info(f"Matched {len(matched_files)} files. Initiating in-memory extraction...")

            # 3. High-Speed Disk I/O into Pandas
            for file_name in matched_files:
                # z.open() reads the file strictly in RAM.
                with z.open(file_name) as f:
                    df = pd.read_csv(f, parse_dates=['date'])
                    # Store in our dictionary using the Scheme Code as the key
                    scheme_code = str(df['Scheme_Code'].iloc[0])
                    fund_data_dict[scheme_code] = df

        logging.info("Data ingestion complete.")
        return fund_data_dict
