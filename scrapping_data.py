##
##       Hulda Nana Nkwenkeu
##   Copyright -  all right reserved - @2021
##

##
## scrapping the real estate website - Zoopla
##


##
##  Loading some required packages
##

from bs4 import BeautifulSoup
from requests import get
import re
import time
import random
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

headers = ({'User-Agent':
            'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.36'})
    

n_page = 0
n_page_max = 450
web_srch_req_root = "https://www.zoopla.co.uk/for-sale/property/london/?q=London&results_sort=newest_listings&search_source=home"
prices = []  ## houses prices
prices_guidance=[]  ## prices guidances such as "In the region of..", "Starting from"
listingIDs = []  ## unique listing ID
iMediaVideoItem = [] ## number of video
iMediaImagesItem = [] ## number of images
iMediaFloorPlanItem = [] ## number of floorplan
iNbOfBedsItems = []  ## numbers of beds
iNbOfBathItems= []  ## numbers of bathrooms
apts_description = []  ## description of the apartment
apts_location = []  ## location of apartment
apts_postCode = [] ## post code of our apartment
transport_information = [] ## all transport close by
iNbOfTransport = []  ## counting the number of transport links
publication_date = [] ## date on which the post was published

for page in range(1,n_page_max):
    web_srch_req = web_srch_req_root + '&pn=' + str(page)
    response = get(web_srch_req, headers = headers)

    #print(response)
    #print(html_soup.prettify())

    html_soup = BeautifulSoup(response.text, 'html.parser')
    regex = re.compile('listing_')
    house_containers = html_soup.find_all('div',{'id' : regex})
    
    
    if house_containers != []:
        
        for container in house_containers:
            
            ##
            ## saving listings ID
            ##
            
            listingIDs.append(container.get('id'))
            
            
            ##
            ## price information extraction
            ##
            regex_price = re.compile('PriceContainer')
            container_data_price = container.find_all('div',{'class':regex_price})
            ### We know there is only going to be a single price container so no need to iterate over the result
            container_data_price_spec = container_data_price[0].find_all('p')
            
            if container_data_price_spec[0].text[0] == '£' or container_data_price_spec[0].text == 'POA' :
                price = container_data_price_spec[0].text
                price_guidance = ""
            else:
                price = container_data_price_spec[1].text
                price_guidance = container_data_price_spec[0].text
                
                
            prices.append(price)
            prices_guidance.append(price_guidance)
            
            ##
            ## Media link containers
            ##
            regex_mediaLink = re.compile('MediaLinksContainer')
            container_data_media = container.find_all('div', {'class':regex_mediaLink})
            ## we know there is only going to be a single media link data
            
            ## number of video
            container_data_media_video = container_data_media[0].find_all('span',{'data-testid':'play'})
            if container_data_media_video == []:
                iNbVideo = 0
            else:
                iNbVideo = container_data_media_video[0].parent.text
            
            iMediaVideoItem.append(iNbVideo)
            
            ## number of images
            container_data_media_images = container_data_media[0].find_all('span',{'data-testid':'camera'})
            if container_data_media_images == []:
                iNbImages = 0
            else:
                iNbImages = container_data_media_images[0].parent.text
            
            iMediaImagesItem.append(iNbImages)
            
            ## number of floorplan
            container_data_media_floorplan = container_data_media[0].find_all('span',{'data-testid':'floorplan'})
            if container_data_media_floorplan == []:
                iNbFloorplan = 0
            else:
                iNbFloorplan = container_data_media_floorplan[0].parent.text
            
            iMediaFloorPlanItem.append(iNbFloorplan)
            
            
            ##
            ##  wrapper features
            ##
            regex_house_features = re.compile('WrapperFeatures')
            container_data_house_features =  container.find_all('div',{'class':regex_house_features})
            ## we know there is only going to be a single house feature class
            
            ## number of beds
            container_data_house_bed = container_data_house_features[0].find_all('span',{'data-testid':'bed'})
            if container_data_house_bed == []:
                iNbOfBeds = ""
            else:
                container_data_house_bed_info = container_data_house_bed[0].parent.parent.find_all('p')
                iNbOfBeds = container_data_house_bed_info[0].text
            
            iNbOfBedsItems.append(iNbOfBeds)
            
            
            ## number of baths
            container_data_house_bath = container_data_house_features[0].find_all('span',{'data-testid':'bath'})
            if container_data_house_bath == []:
                iNbOfBaths = ""
            else:
                container_data_house_bath_info = container_data_house_bath[0].parent.parent.find_all('p')
                iNbOfBaths = container_data_house_bed_info[0].text
            
            iNbOfBathItems.append(iNbOfBaths)
            
            
            ##
            ##  appartement location
            ##
            
            regex_house_location = re.compile('listing-details-link')
            container_data_house_location = container.find_all('a',{'data-testid':regex_house_location})
            # we know there is only going to be one item
            
            ## apartment description
            container_data_house_location_description = container_data_house_location[0].find_all('h2')
            apt_description = container_data_house_location_description[0].text
            apts_description.append(apt_description)
            
            
            # apartment location
            container_data_house_location_postCode = container_data_house_location[0].find_all('p')
            apt_post_code = container_data_house_location_postCode[0].text
            apts_location.append(apt_post_code)
            
            # extracting postcode from location
            # we know it is always at the end of the string
            last_space_pos = apt_post_code.rfind(" ") + 1
            final_post_code = apt_post_code[last_space_pos:len(apt_post_code)]
            apts_postCode.append(final_post_code)
            
            ##
            ## transport information
            ##
            
            regex_transport = re.compile('TransportWrapper')
            container_data_transport = container.find_all('div', {'class':regex_transport})
            # we know there is only going to be one
            
            container_data_transport_info = container_data_transport[0].find_all('p')
            it_transport = 0
            transport_link = ""
            for container_datat_transport_info_spec in container_data_transport_info:
                if it_transport == 0:
                    transport_link = container_datat_transport_info_spec.text
                else:
                    transport_link = transport_link + ";" + container_datat_transport_info_spec.text
                    
                it_transport = it_transport + 1
                    
            transport_information.append(transport_link)
            iNbOfTransport.append(it_transport)      
            
            
            ##
            ## publication date
            ##
            
            regex_date_listed = re.compile('CardFooter')
            container_data_date_listed = container.find_all('div',{'class': regex_date_listed})
            # we know it is only going be to be a single one
            container_data_date_listed_spec = container_data_date_listed[0].find_all('span',{'data-testid':'date-published'})
            pub_date = container_data_date_listed_spec[0].text
            pub_date = pub_date.replace('Listed on ', '')
            publication_date.append(pub_date)
            
    else:
        break
    
    time.sleep(random.randint(1,2))
    n_page = n_page + 1


print('You scraped {} pages containing {} properties.'.format(n_page, len(prices)))

cols = ['Price','Price Guidance','Listing ID','Media Video','Media Images','Media Floor Plan','Number of Beds','Number of Baths','Apt Description','Apt Location','Apt Post Code','Transport Information','Number of Transport','Publication Date']

Zoopla_data = pd.DataFrame({'Price' : prices,
                            'Price Guidance' : prices_guidance,
                            'Listing ID' : listingIDs,
                            'Media Video' : iMediaVideoItem,
                            'Media Images' : iMediaImagesItem,
                            'Media Floor Plan' : iMediaFloorPlanItem,
                            'Number of Beds' : iNbOfBedsItems,
                            'Number of Baths' : iNbOfBathItems,
                            'Apt Description' : apts_description,
                            'Apt Location' : apts_location,
                            'Apt Post Code' : apts_postCode,
                            'Transport Information' : transport_information,
                            'Number of Transport' : iNbOfTransport,
                            'Publication Date' : publication_date})[cols]

Zoopla_data.to_excel('Zoopla_data_info.xls')
