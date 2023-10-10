import tools.building_seg as tbs
import os
import os.path as osp

def execute():
    # 原始影像数据，解压目录
    local_data_dir = os.environ.get('LOCAL_DATA_DIR')
    # 数据处理结果文件存放目录
    local_output_dir = os.environ.get('LOCAL_OUTPUT_DIR')
    extract(local_data_dir,local_output_dir)

def extract(data_dir,result_out_dir): 
    data_name = osp.split(data_dir)[-1]
    out_dir = osp.join(result_out_dir,data_name)
    img_list = os.listdir(data_dir)
    for img in img_list:
        if (img.lower().endswith(('.tif','.tiff'))):
            img_path = osp.join(data_dir,img)
            tbs.extract_data(image_path=img_path,save_dir=out_dir,block_size=512,overlap=36)



if __name__ == "__main__":
    datadir = '/home/zju/data_szw/test/data/austin7'
    outdir = '/home/zju/data_szw/test/output'
    extract(datadir,outdir)