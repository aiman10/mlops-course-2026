import re
import pandas as pd
from sklearn.impute import SimpleImputer

class Cleaner:
    def __init__(self):
        self.imputer = SimpleImputer(strategy='most_frequent')

    def clean_data(self, data):
        data = data.drop(['id', 'SalesChannelID', 'VehicleAge', 'DaysSinceCreated'], axis=1)

        # Handle £ currency format
        data['AnnualPremium'] = data['AnnualPremium'].apply(
            lambda x: float(re.sub(r'[£$,\s]', '', str(x)))
        )

        data[['Gender', 'RegionID']] = self.imputer.fit_transform(data[['Gender', 'RegionID']])
        data['Age'] = data['Age'].fillna(data['Age'].median())
        data['HasDrivingLicense'] = data['HasDrivingLicense'].fillna(1)
        data['Switch'] = data['Switch'].fillna(-1)
        data['PastAccident'] = data['PastAccident'].fillna('Unknown')

        Q3 = data['AnnualPremium'].quantile(0.75)
        IQR = data['AnnualPremium'].quantile(0.75) - data['AnnualPremium'].quantile(0.25)
        upper_bound = Q3 + 1.5 * IQR
        data = data[data['AnnualPremium'] <= upper_bound]

        return data
