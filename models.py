import os
import urllib.request as urllib2
from bs4 import BeautifulSoup

url_format = 'https://www.legis.state.tx.us/BillLookup/History.aspx?LegSess={legis_session}R&Bill=' \
                'HB{bill_num}'

file_name = 'labels.txt'
file = open(file_name, 'w')
number_bills = 4765

def house_bill_text(legis_session: int, bill_num: str) -> str:
    """
    :param legis_session:
    :param bill_num:
    :param bill_type:
    :return:
    """

    bill_url = url_format.format(legis_session=legis_session, bill_num=bill_num)

    page = urllib2.urlopen(bill_url)
    soup = BeautifulSoup(page, 'html.parser')
    committee_name = soup.find('td', attrs={'id': 'cellComm1Committee'})


    return committee_name.a.text


for i in range(1, number_bills):

    committee_name = house_bill_text(86, i)
    file.write(committee_name + '\n') # pad bill number to 5-digits
