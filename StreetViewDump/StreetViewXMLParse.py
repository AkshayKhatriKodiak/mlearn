#!/usr/bin/env python
# -*- coding: utf-8 -*- 

import xml.etree.ElementTree as et
from collections import namedtuple

def streetViewXmlParse(s):
	latitude = None
	longitude = None
	pano_id = None
	linkids = list()
	try:
		xml = et.fromstring(s)
		DataTup = namedtuple("DataTup", "latitude longitude pano_id linkids")
		data_properties = xml.find("data_properties").attrib

		latitude = float(data_properties['lat'])
		longitude = float(data_properties['lng'])
		pano_id = data_properties['pano_id']

		links = xml.find('annotation_properties').findall('link')
		for link in links:
			linkids.append(link.attrib['pano_id'])

		return DataTup(float(latitude), float(longitude),
			pano_id, linkids)

	except (et.ParseError, AttributeError) as e:
		print("XML \n%s\ncaused error %s\n" % (s, repr(e)))
		return None 

def test():
	string = """
				<panorama>
				<data_properties image_width="13312" image_height="6656" 
				tile_width="512" tile_height="512" image_date="2011-05" 
				pano_id="epVrmnyWrms1qjgnM9Cwew" imagery_type="1" num_zoom_levels="5" 
				lat="37.365371" lng="-122.142095" original_lat="37.365354" 
				original_lng="-122.142109" elevation_wgs84_m="123.534105" 
				elevation_egm96_m="155.796829">
				<copyright>© 2016 Google</copyright>
				<text/>
				<region>Los Altos Hills, California</region>
				<country>United States</country>
				</data_properties>
				<projection_properties projection_type="spherical"
				pano_yaw_deg="155.58" tilt_yaw_deg="16.41" tilt_pitch_deg="3.62"/>
				<annotation_properties>
				<link yaw_deg="151.61" pano_id="iCQC-8S1hVlhkL2DSyRtlQ" road_argb="0x80fdf872">
				<link_text/>
				</link>
				<link yaw_deg="118.53" pano_id="OzqgS-D9UFiKlWNb-79K6A" road_argb="0x80fdf872">
				<link_text/>
				</link>
				<link yaw_deg="334.72" pano_id="Zw0smx7y-5_FUCTwvzxHOQ" road_argb="0x80fdf872">
				<link_text/>
				</link>
				</annotation_properties>
				</panorama>
			"""

	print streetViewXmlParse(string)

if __name__ == "__main__":
	test()
