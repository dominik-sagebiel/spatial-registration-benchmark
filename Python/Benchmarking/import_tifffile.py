import tifffile
with tifffile.TiffFile(r"C:\Users\domin\Nextcloud\Uni\FU Ba. BioInf\11. Sem\Bachelorarbeit\data\mIF\LymphNode\adjacent\Core_13.ome.tif") as tif:
    print(tif.series[0].axes, tif.series[0].shape)

