# %%
from urllib.request import urlopen
from bs4 import BeautifulSoup
from collections import namedtuple

Item = namedtuple('Item', ["id", "gender", "age", "diseases"])
url = "https://koronavirus.gov.hu/elhunytak?page="
# %%
def read_all() -> list[Item]:
    page = 0
    items = []
    while True:
        if (page % 10) == 0:
            print("Loading page ", page)
        html: str = urlopen(url + str(page)).read().decode("utf-8")
        soup = BeautifulSoup(html, "html.parser")
        rows = soup.find_all("tr")
        if len(rows) == 0:
            return items
        for row in rows[1:]:
            cols = row.find_all("td")
            item = Item(*[col.get_text(strip=True) for col in cols])
            items.append(item)
        page += 1
items = read_all()
# %%
import csv
f = open('data.csv', 'w', encoding="UTF8", newline='')
writer = csv.writer(f)

writer.writerow(["id", "gender", "age", "diseases"])
for item in items:
    writer.writerow([item.id, item.gender, item.age, item.diseases])

f.close()
print("DONE")
# %%
