import requests as r
from bs4 import BeautifulSoup

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


