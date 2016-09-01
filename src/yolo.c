#include "network.h"
#include "detection_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "demo.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#ifdef OPENCV
#include "opencv2/highgui/highgui_c.h"
#endif


/* Change class number here */
#define CLASSNUM 1

/* Change class names here */
char *voc_names[] = {"Hole"};
image voc_labels[CLASSNUM];

void train_yolo(char *cfgfile, char *weightfile, char* train_images, char* backup_directory)
{
	//char *train_images = "/data/voc/train.txt";
	//char *backup_directory = "/home/pjreddie/backup/";
	srand(time(0));
	data_seed = time(0);
	char *base = basecfg(cfgfile);
	printf("%s\n", base);
	float avg_loss = -1;
	network net = parse_network_cfg(cfgfile);
	if(weightfile){
		load_weights(&net, weightfile);
	}
	printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
	int imgs = net.batch*net.subdivisions;
	int i = *net.seen/imgs;
	data train, buffer;


	layer l = net.layers[net.n - 1];

	//int side = l.side;
	int rows = l.rows;
	int cols = l.cols;
	int classes = l.classes;
	float jitter = l.jitter;

	list *plist = get_paths(train_images);
	//int N = plist->size;
	char **paths = (char **)list_to_array(plist);

	load_args args = {0};
	args.w = net.w;
	args.h = net.h;
	args.channels = net.c;
	args.paths = paths;
	args.n = imgs;
	args.m = plist->size;
	args.classes = classes;
	args.jitter = jitter;
	args.num_rows = rows;
	args.num_cols = cols;
	args.d = &buffer;
	args.type = REGION_DATA;

	pthread_t load_thread = load_data_in_thread(args);
	clock_t time;
	//while(i*imgs < N*120){
	while(get_current_batch(net) < net.max_batches){
		i += 1;
		time=clock();
		pthread_join(load_thread, 0);
		train = buffer;
		load_thread = load_data_in_thread(args);

		printf("Loaded: %lf seconds\n", sec(clock()-time));

		time=clock();
		float loss = train_network(net, train);
		if (avg_loss < 0) avg_loss = loss;
		avg_loss = avg_loss*.9 + loss*.1;

		printf("%d: %f, %f avg, %f rate, %lf seconds, %d images\n", i, loss, avg_loss, get_current_rate(net), sec(clock()-time), i*imgs);
		if(i%1000==0 || (i < 1000 && i%500 == 0)){
			char buff[256];
			sprintf(buff, "%s/%s_%d.weights", backup_directory, base, i);
			save_weights(net, buff);
		}
		free_data(train);
	}
	char buff[256];
	sprintf(buff, "%s/%s_final.weights", backup_directory, base);
	save_weights(net, buff);
}

void convert_detections(float *predictions, int classes, int num, int square, int rows, int cols, int w, int h, float thresh, float **probs, box *boxes, int only_objectness)
{
	int i,j,n;
	//int per_cell = 5*num+classes;
	for (i = 0; i < rows*cols; ++i){
		int row = i / cols;
		int col = i % cols;
		for(n = 0; n < num; ++n){
			int index = i*num + n;
			int p_index = rows*cols*classes + i*num + n;
			float scale = predictions[p_index];
			int box_index = rows*cols*(classes + num) + (i*num + n)*4;
			boxes[index].x = (predictions[box_index + 0] + col) / cols * w;
			boxes[index].y = (predictions[box_index + 1] + row) / rows * h;
			boxes[index].w = pow(predictions[box_index + 2], (square?2:1)) * w;
			boxes[index].h = pow(predictions[box_index + 3], (square?2:1)) * h;
			for(j = 0; j < classes; ++j){
				int class_index = i*classes;
				float prob = scale*predictions[class_index+j];
				probs[index][j] = (prob > thresh) ? prob : 0;
			}
			
			if(only_objectness){
				probs[index][0] = scale;
			}
		}
	}
}

void print_yolo_detections(FILE **fps, char *id, box *boxes, float **probs, int total, int classes, int w, int h)
{
	int i, j;
	for(i = 0; i < total; ++i){
		float xmin = boxes[i].x - boxes[i].w/2.;
		float xmax = boxes[i].x + boxes[i].w/2.;
		float ymin = boxes[i].y - boxes[i].h/2.;
		float ymax = boxes[i].y + boxes[i].h/2.;

		if (xmin < 0) xmin = 0;
		if (ymin < 0) ymin = 0;
		if (xmax > w) xmax = w;
		if (ymax > h) ymax = h;

		for(j = 0; j < classes; ++j){
			if (probs[i][j]) fprintf(fps[j], "%s %f %f %f %f %f\n", id, probs[i][j],
					xmin, ymin, xmax, ymax);
		}
	}
}

void validate_yolo(char *cfgfile, char *weightfile, char *val_images, char *result_dir)
{
	network net = parse_network_cfg(cfgfile);
	if(weightfile){
		load_weights(&net, weightfile);
	}
	set_batch_network(&net, 1);
	fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
	srand(time(0));

	//create output directory if it does not exist
	struct stat st= {0};
	if(stat(result_dir,&st)==-1){
		fprintf(stderr,"Creating output directory\n");
		mkdir(result_dir,0700);
	}

	char *base = result_dir;
	list *plist = get_paths(val_images);
	char **paths = (char **)list_to_array(plist);

	layer l = net.layers[net.n-1];
	int classes = l.classes;
	int square = l.sqrt;
	//int side = l.side;
	int rows = l.rows;
	int cols = l.cols;

	int j;
	FILE **fps = calloc(classes, sizeof(FILE *));
	for(j = 0; j < classes; ++j){
		char buff[1024];
		snprintf(buff, 1024, "%s%s.txt", base, voc_names[j]);
		fps[j] = fopen(buff, "w");
	}
	box *boxes = calloc(rows*cols*l.n, sizeof(box));
	float **probs = calloc(rows*cols*l.n, sizeof(float *));
	for(j = 0; j < rows*cols*l.n; ++j) probs[j] = calloc(classes, sizeof(float *));

	int m = plist->size;
	int i=0;
	int t;

	float thresh = .001;
	int nms = 1;
	float iou_thresh = .5;

	int nthreads = 2;
	image *val = calloc(nthreads, sizeof(image));
	image *val_resized = calloc(nthreads, sizeof(image));
	image *buf = calloc(nthreads, sizeof(image));
	image *buf_resized = calloc(nthreads, sizeof(image));
	pthread_t *thr = calloc(nthreads, sizeof(pthread_t));

	load_args args = {0};
	args.w = net.w;
	args.h = net.h;
	args.type = IMAGE_DATA;

	for(t = 0; t < nthreads; ++t){
		args.path = paths[i+t];
		args.im = &buf[t];
		args.resized = &buf_resized[t];
		thr[t] = load_data_in_thread(args);
	}
	time_t start = time(0);
	for(i = nthreads; i < m+nthreads; i += nthreads){
		fprintf(stderr, "%d\n", i);
		for(t = 0; t < nthreads && i+t-nthreads < m; ++t){
			pthread_join(thr[t], 0);
			val[t] = buf[t];
			val_resized[t] = buf_resized[t];
		}
		for(t = 0; t < nthreads && i+t < m; ++t){
			args.path = paths[i+t];
			args.im = &buf[t];
			args.resized = &buf_resized[t];
			thr[t] = load_data_in_thread(args);
		}
		for(t = 0; t < nthreads && i+t-nthreads < m; ++t){
			char *path = paths[i+t-nthreads];
			char *id = basecfg(path);
			float *X = val_resized[t].data;
			float *predictions = network_predict(net, X);
			int w = val[t].w;
			int h = val[t].h;
			convert_detections(predictions, classes, l.n, square, rows, cols, w, h, thresh, probs, boxes, 0);
			if (nms) do_nms_sort(boxes, probs, rows*cols*l.n, classes, iou_thresh);
			print_yolo_detections(fps, id, boxes, probs, rows*cols*l.n, classes, w, h);
			free(id);
			free_image(val[t]);
			free_image(val_resized[t]);
		}
	}
	fprintf(stderr, "Total Detection Time: %f Seconds\n", (double)(time(0) - start));
}

void validate_yolo_recall(char *cfgfile, char *weightfile, char *val_images, char *out_dir, float th)
{
	network net = parse_network_cfg(cfgfile);
	if(weightfile){
		load_weights(&net, weightfile);
	}
	set_batch_network(&net, 1);
	fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
	srand(time(0));

	//create output directory if it does not exist
	struct stat st= {0};
	if(stat(out_dir,&st)==-1){
		fprintf(stderr,"Creating output directory\n");
		mkdir(out_dir,0700);
	}

	char *base = out_dir;
	list *plist = get_paths(val_images);
	char **paths = (char **)list_to_array(plist);

	layer l = net.layers[net.n-1];
	int classes = l.classes;
	int square = l.sqrt;
	//int side = l.side;
	int rows = l.rows;
	int cols = l.cols;

	int j, k;
	FILE **fps = calloc(classes, sizeof(FILE *));
	for(j = 0; j < classes; ++j){
		char buff[1024];
		snprintf(buff, 1024, "%s%s.txt", base, voc_names[j]);
		fps[j] = fopen(buff, "w");
	}
	box *boxes = calloc(rows*cols*l.n, sizeof(box));
	float **probs = calloc(rows*cols*l.n, sizeof(float *));
	for(j = 0; j < rows*cols*l.n; ++j) probs[j] = calloc(classes, sizeof(float *));

	int m = plist->size;
	int i=0;

	float thresh = th;
	float iou_thresh[11] = {0.0,0.05,0.1,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.5};
	float nms = 0;

	int total = 0;
	int correct[11] = {0,0,0,0,0,0,0,0,0,0,0};
	int proposals = 0;
	float avg_iou = 0;
	Vector id_found;
	initArray(&id_found,1);

	for(i = 0; i < m; ++i){
		char *path = paths[i];
		image orig = load_image(path, 0, 0,net.c);
		image sized = resize_image(orig, net.w, net.h);
		char *id = basecfg(path);
		float *predictions = network_predict(net, sized.data);
		convert_detections(predictions, classes, l.n, square, rows, cols, 1, 1, thresh, probs, boxes, 0);
		if (nms) do_nms(boxes, probs, rows*cols*l.n, 1, nms);

		char *labelpath = find_replace(path, "images", "labels");
		labelpath = find_replace(labelpath, "JPEGImages", "labels");
		labelpath = find_replace(labelpath, ".jpg", ".txt");
		labelpath = find_replace(labelpath, ".JPEG", ".txt");
		labelpath = find_replace(labelpath, ".png", ".txt");
		labelpath = find_replace(labelpath, ".PNG", ".txt");

		int num_labels = 0;
		box_label *truth = read_boxes(labelpath, &num_labels);
		for(k = 0; k < rows*cols*l.n; ++k){
			if(probs[k][0] > thresh){
				++proposals;
			}
		}
		for (j = 0; j < num_labels; ++j) {
			++total;
			while(id_found.used < truth[j].id)
				insertArray(&id_found,0);
			box t = {truth[j].x, truth[j].y, truth[j].w, truth[j].h};
			float best_iou = 0;
			for(k = 0; k < rows*cols*l.n; ++k){
				float iou = box_iou(boxes[k], t);
				//find overlapping prediction
				if(iou > best_iou){
					//find the predicted class
					float best_score = thresh;
					int best_class_index = -1;
					for(int c=0; c<CLASSNUM; c++){
						if(probs[k][c]>best_score){
							best_score = probs[k][c];
							best_class_index = c;
						}
					}
					//check if it's good or not
					if(best_class_index == truth[j].classe)
						best_iou = iou;
				}
			}
			avg_iou += best_iou;
			for(int k=0; k<11; k++){
				if(best_iou > iou_thresh[k]){
					id_found.array[truth[j].id]=1;
					++correct[k];
				}
			}
		}
		if(i%10==0){
			printf("\033[2J");
			printf("\033[1;1H");
			printf("#img\tPred\tTP\ttot\tRPs/Img\tAvg-IOU\tRecall\tPrecision\n");
			printf("%5d\t%5d\t%5d\t%5d\t%.2f\t%.2f%%\t%.2f%%\t%.2f%%\n", i, proposals, correct[10], total, (float)proposals/(i+1), avg_iou*100/total, 100.*correct[10]/total, 100.*correct[10]/proposals);
			printf("IOU_th\tTP\tFP\tRecall\tPrecision\n");
			for(int k=0; k<11; k++){
				printf("%.2f%%\t%5d\t%5d\t%.2f%%\t%.2f%%\t\n", iou_thresh[k], correct[k], proposals-correct[k], 100.*correct[k]/total, 100.*correct[k]/proposals);
			}
			int found=0;
			for(int i=0; i<=id_found.used; i++)
				found+=id_found.array[i];
			printf("Founded: %d/%d\n", found, id_found.used+1);
		}
		free(id);
		free_image(orig);
		free_image(sized);
	}
	for(j = 0; j < classes; ++j){
		fprintf(fps[j],"IOU_th;TP;FP;Recall;Precision\n");
		for(int k=0; k<11; k++){
			fprintf(fps[j],"%.2f%%;%5d;%5d;%.2f%%;%.2f%%;\n", iou_thresh[k], correct[k], proposals-correct[k], 100.*correct[k]/total, 100.*correct[k]/proposals);
		}
		fprintf(fps[j], "\n\nFounded;Total;\n");
		int found=0;
		for(int i=0; i<=id_found.used; i++)
			found+=id_found.array[i];
		fprintf(fps[j], "%d;%d;\n", found, id_found.used);
	}
	freeArray(&id_found);
}

void test_yolo(char *cfgfile, char *weightfile, char *filename, float thresh)
{

	network net = parse_network_cfg(cfgfile);
	if(weightfile){
		load_weights(&net, weightfile);
	}
	detection_layer l = net.layers[net.n-1];
	set_batch_network(&net, 1);
	srand(2222222);
	clock_t time;
	char buff[256];
	char *input = buff;
	int j;
	float nms=.5;
	box *boxes = calloc(l.rows*l.cols*l.n, sizeof(box));
	float **probs = calloc(l.rows*l.cols*l.n, sizeof(float *));
	for(j = 0; j < l.rows*l.cols*l.n; ++j) probs[j] = calloc(l.classes, sizeof(float *));
	while(1){
		if(filename){
			strncpy(input, filename, 256);
		} else {
			printf("Enter Image Path: ");
			fflush(stdout);
			input = fgets(input, 256, stdin);
			if(!input) return;
			strtok(input, "\n");
		}
		image im = load_image(input,0,0,net.c);
		image sized = resize_image(im, net.w, net.h);
		float *X = sized.data;
		time=clock();
		float *predictions = network_predict(net, X);
		printf("%s: Predicted in %f seconds.\n", input, sec(clock()-time));
		convert_detections(predictions, l.classes, l.n, l.sqrt, l.rows, l.cols, 1, 1, thresh, probs, boxes, 0);
		if (nms) do_nms_sort(boxes, probs, l.rows*l.cols*l.n, l.classes, nms);
		//draw_detections(im, l.side*l.side*l.n, thresh, boxes, probs, voc_names, voc_labels, 20);
		draw_detections(im, l.rows*l.cols*l.n, thresh, boxes, probs, voc_names, voc_labels, CLASSNUM);
		save_image(im, "predictions");
		show_image(im, "predictions");

		show_image(sized, "resized");
		free_image(im);
		free_image(sized);
#ifdef OPENCV
		cvWaitKey(0);
		cvDestroyAllWindows();
#endif
		if (filename) break;
	}
}

void run_yolo(int argc, char **argv)
{
	int i;
	for(i = 0; i < CLASSNUM; ++i){
		char buff[256];
		sprintf(buff, "data/labels/%s.png", voc_names[i]);
		voc_labels[i] = load_image_color(buff, 0, 0);
	}

	float thresh = find_float_arg(argc, argv, "-thresh", .2);
	int cam_index = find_int_arg(argc, argv, "-c", 0);
	int frame_skip = find_int_arg(argc, argv, "-s", 0);
	int save_video = find_int_arg(argc,argv,"-save",1);
	if(argc < 4){
		fprintf(stderr, "usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", argv[0], argv[1]);
		return;
	}

	char *cfg = argv[3];
	char *weights = (argc > 4) ? argv[4] : 0;
	char *filename = (argc > 5) ? argv[5]: 0;
	if(0==strcmp(argv[2], "test")) test_yolo(cfg, weights, filename, thresh);
	else if(0==strcmp(argv[2], "train")){
		if(argc>6){
			char *train_images_txt = argv[4];
			char *backup_directory = argv[5];
			weights = argv[6];
			train_yolo(cfg, weights, train_images_txt, backup_directory);
		} else {
			fprintf(stderr, "usage: %s %s [train] [cfg] [train_images_txt] [backup_directory] [weights (optional)]\n", argv[0], argv[1]);
			return;
		}
	}
	else if(0==strcmp(argv[2], "valid")){ 
		if(argc>6){
			char *val_images_txt = argv[4];
			char *out_directory = argv[5];
			weights = argv[6];
			validate_yolo(cfg, weights,val_images_txt,out_directory);
		} else {
			fprintf(stderr, "usage: %s %s [valid] [cfg] [val_images_txt] [out_directory] [weights (optional)]\n", argv[0], argv[1]);
			return;
		}
	}
	else if(0==strcmp(argv[2], "recall")){
		if(argc>6){
			char *val_images_txt = argv[4];
			char *out_directory = argv[5];
			weights = argv[6];
			validate_yolo_recall(cfg, weights, val_images_txt, out_directory, thresh);
		} else {
			fprintf(stderr, "usage: %s %s [recall] [cfg] [val_images_txt] [out_directory] [weights (optional)]\n", argv[0], argv[1]);
			return;
		}
	}
	else if(0==strcmp(argv[2], "demo")) demo(cfg, weights, thresh, cam_index, filename, voc_names, voc_labels, CLASSNUM, frame_skip, save_video);
}
