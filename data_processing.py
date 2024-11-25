import requests as r
from bs4 import BeautifulSoup
import csv
import copy
import misc
import numpy as np
import pandas as pd
import re

def get_state_list():
    f = r.get("https://www.archives.gov/electoral-college/2024")
    html_content = f.content
    soup = BeautifulSoup(html_content, 'html.parser')
    table = [table for table in soup.find_all("table")][1]
    states_list = []
    for _ in table.find_all("tr")[2:]:
        states_list += [_.find("td").text]
    return states_list[:-2]

def get_president(n):
#n: past n elections.
    _ = []
    for i in range(n):
        f = r.get("https://www.archives.gov/electoral-college/" + str(2024 - i*4))
        html_content = f.content
        soup = BeautifulSoup(html_content, 'html.parser')
        table = [table for table in soup.find_all("table")][0]
        _ += [table.find_all("tr")[0].find("td").text]
    return _

def get_county_results(state, year):
    f = r.get("https://en.wikipedia.org/wiki/" + str(year) + "_United_States_presidential_election_in_" + str(state) + "#By_county")
    html_content = f.content
    soup = BeautifulSoup(html_content, 'html.parser')

    table = None
    for _ in soup.find_all(True):
        if(_.name == "h3" and ("county" in _.text or "County" in _.text)):
            for s in _.parent.next_siblings:
                if(s.name == "table"):
                    table = s
                    break
    try:
        counties = table.find_all("tr")[2:-1]
    except Exception as e:
        pass
    results = []
  
    for _ in counties:
        try:
            result = {}
            county = _.find("td")
            county_color = county.attrs["style"].split("background-color:")[1].split(";")[0]
            if county_color == "#FFB6B6":
                county_party = 'R'
            else:
                county_party = 'D'
            county_name = county.text.strip("\n")
            result["State"] = state
            result["Year"] = year
            result["Party"] = county_party
            result["County"] = county_name
            results += [result]
        except Exception as e:
            pass
    return results


def get_past_ten_elections_by_county(state):
    results = []
    for i in range (10):
        try:
            results += get_county_results(state, 2008 - 4*i)
        except Exception as e:
            print(f"An error occurred: {e}")
            print(state, 2008 - 4*i)
    return results

def get_past_ten_elections_by_county_all_states():
    states = get_state_list()
    results = []
    for state in states:
        results += get_past_ten_elections_by_county(state)

    with open('by_county.csv', 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['State', 'Year', 'County', 'Party'])
        writer.writeheader()
        writer.writerows(results)
    return results

def get_training_data():
    results = read_data('by_county.csv')

    # MERGING CAINC DARTA WITH BY COUNTY ELECTION RESULTS.
    for result in results:
        state_abbrev = misc.us_state_to_abbrev[result["State"]]
        try:
            with open('CAINC30_Economic Profile by County/CAINC30_'+ state_abbrev + '_1969_2023.csv', 'r') as file:
                csv_reader = csv.DictReader(file)
                data = list(csv_reader)[:-5]
            for _ in data:
                del _["GeoFIPS"]
                del _["Region"]
                del _["TableName"]
                del _["Unit"]
                del _["LineCode"]
            
            
            econ_data = list(filter(lambda d: result["County"] in d["GeoName"], data[:-5]))
            
            for ed in econ_data:
                year = int(result["Year"])
                label = ed["Description"]
                result[label + "YR-1"] = int(ed[str(year - 1)]) 
                result[label + "YR-2"] = int(ed[str(year - 2)])
                result[label + "YR-3"] = int(ed[str(year - 3)])
        except Exception as e:
            print(e)
            pass

    for result in results:
        state_abbrev = misc.us_state_to_abbrev[result["State"]]
        try:
            with open('CAINC1_Annual Personal Income by County/CAINC1_'+ state_abbrev + '_1969_2023.csv', 'r') as file:
                csv_reader = csv.DictReader(file)
                data = list(csv_reader)[:-5]
            for _ in data:
                del _["GeoFIPS"]
                del _["Region"]
                del _["TableName"]
                del _["Unit"]
                del _["LineCode"]
            
            
            econ_data = list(filter(lambda d: result["County"] in d["GeoName"], data[:-5]))
            
            for ed in econ_data:
                year = int(result["Year"])
                label = ed["Description"]
                result[label + "YR-1"] = int(ed[str(year - 1)]) 
                result[label + "YR-2"] = int(ed[str(year - 2)])
                result[label + "YR-3"] = int(ed[str(year - 3)])
        except Exception as e:
            print(e)
            pass
    
        # REMOVE ROWS WITH TOO MANY MISSING DATA
        data = list(filter(lambda d: len(d.keys()) == len(list(filter(None, list(d.values())))) , results))

        # ADJUST FOR INFLATION USING PANDAS
        df = pd.DataFrame(data)
        # remove population columns
        columns = df.columns
        populations = [df['Population (persons) 1/YR-1'],
                    df['Population (persons) 1/YR-2'],
                    df['Population (persons) 1/YR-3']]
        df.drop(columns = "Population (persons) 1/YR-1", inplace=True)
        df.drop(columns = "Population (persons) 1/YR-2", inplace=True)
        df.drop(columns = "Population (persons) 1/YR-3", inplace=True)
        df.drop(columns = " Population (persons) 3/YR-1", inplace=True)
        df.drop(columns = " Population (persons) 3/YR-2", inplace=True)
        df.drop(columns = " Population (persons) 3/YR-3", inplace=True)
        #adjust for inflation
        for i in df.index:
            try:
                year = int(df.iloc[i,1])
                df.iloc[i, 4:] = df.iloc[i, 4:].map(lambda x: int(x) * misc.cpi_dict[year])
            except Exception as e:
                print(e)
                pass

        #add population columns back
        df["Population YR-1"] = populations[0]
        df["Population YR-2"] = populations[1]
        df["Population YR-3"] = populations[2]

        # NORMALIZE DATA
        for column in df.columns:
            try:
                if column != "Year":
                    df[column] = df[column].map(lambda x: int(x))
                    x_max = max(list(df[column]))
                    x_min = min(list(df[column]))
                    df[column] = df[column].map(lambda x: (x - x_min)/(x_max - x_min))
            except Exception as e:
                print(e)
        df.to_csv("get_training_data()_output.csv")
        return data
    
    
    


def read_data(name, mode = "Dict"):
    
    with open(name, 'r') as file:
        csv_reader = csv.DictReader(file)
        data = list(csv_reader)
    
    if mode == "List":
        l= []
        target = []
        for d in data:
            l += [list(d.values())[5:]]
            target += [list(d.values())[4]]
        return target, np.array(l, dtype= np.float32)
    return data

def county_vote_csv_to_df(filename):
    """
    change the format  of countypres_2020-2020.csv to pandas dataframe format,
    the title is: year, state, county_name, candidate, candidatevotes

    returns: a pandas dataframe of the csv file.
    """
    data = pd.read_csv(filename)
    result_df = data[['year', 'state', 'county_name',"party", 'candidate', 'candidatevotes']]

    # Sort by year, state, and county
    result_df = result_df.sort_values(['year', 'state', 'county_name', 'candidate'])
    return result_df
