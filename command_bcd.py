import tools.building_cd as tbcd
import os
import os.path as osp

def execute():
    # 原始影像1数据，解压目录
    local_data_dir = os.environ.get('LOCAL_DATA_DIR')
    # 原始影像2数据，解压目录
    local_later_data_dir = os.environ.get('LOCAL_LATER_DATA_DIR')
    # 数据处理结果文件存放目录
    local_output_dir = os.environ.get('LOCAL_OUTPUT_DIR')

    
    extract(local_data_dir,local_later_data_dir,local_output_dir,model_path='/home/other/best_model/')

def extract(data_dir,later_data_dir,result_out_dir,model_path): 
    # data_name = osp.split(data_dir)[-1]
    out_dir = osp.join(result_out_dir,'result')
    img1_file = get_img_file(data_dir)[0]
    img2_file = get_img_file(later_data_dir)[0]
    print("file1:"+img1_file)
    print("file2:"+img2_file)
    tbcd.extract_data(image1_path=img1_file,image2_path=img2_file,save_dir=out_dir,block_size=512,overlap=36,model_path=model_path)

def get_img_file(file_dir):
    img_list = os.listdir(file_dir)
    img_file_list = []
    for img in img_list:
        if (img.lower().endswith(('.tif','.tiff','.png','jpg','jpeg'))):
            img_file_list.append(osp.join(file_dir,img))
    return img_file_list




if __name__ == "__main__":
    image1_path = '/home/zju/data_szw/test/data/cd/img1'
    image2_path = '/home/zju/data_szw/test/data/cd/img2'
    model_path = '/home/zju/lizhu_docker/bcd1.0_bit_whubcd/other/best_model/'
    outdir = '/home/zju/data_szw/test/data/cd/output'
    extract(image1_path,image2_path,outdir,model_path)