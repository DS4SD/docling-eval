
DOCLING_VERSION = "docling_version"
STATUS_COLUMN = "conversion_status"

GROUNDTRUTH_DOC_COLUMN = "GroundTruthDoclingDocument"
DOCUMENT_COLUMN = "DoclingDocument"
BINARY_DOCCOLUMN = "BinaryDocument"

PICTURES_COLUMN = "pictures"
PAGE_IMAGES_COLUMN= "page_images"

HTML_DEFAULT_HEAD: str = r"""<head>
<link rel="icon" type="image/png"
href="https://ds4sd.github.io/docling/assets/logo.png"/>
<meta charset="UTF-8">
<title>
Powered by Docling
</title>
<style>
html {
background-color: LightGray;
}
body {
margin: 0 auto;
width:800px;
padding: 30px;
background-color: White;
font-family: Arial, sans-serif;
box-shadow: 10px 10px 10px grey;
}
figure{
display: block;
width: 100%;
margin: 0px;
margin-top: 10px;
margin-bottom: 10px;
}
img {
display: block;
margin: auto;
margin-top: 10px;
margin-bottom: 10px;
max-width: 640px;
max-height: 640px;
}
table {
min-width:500px;
background-color: White;
border-collapse: collapse;
cell-padding: 5px;
margin: auto;
margin-top: 10px;
margin-bottom: 10px;
}
th, td {
border: 1px solid black;
padding: 8px;
}
th {
font-weight: bold;
}
table tr:nth-child(even) td{
background-color: LightGray;
}
</style>
</head>"""
