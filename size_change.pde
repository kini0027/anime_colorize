String path = "./images1/";

int imageWidth = 256;
int imageHeight = 256;

int margin = 0;

int imageNum = 2;
int a=100;

int originImageWidth;
int originImageHeight;

int canvasWidth;
int canvasHeight;

PImage images[][] = new PImage[a][imageNum];

void setup() {
  for (int j = 0; j < a; j++) {
    for (int i = 0; i < imageNum; i++) {
      images[j][i] = loadImage(path + j + i + ".jpg");
    }
  }

  originImageWidth = images[0][0].width;
  originImageHeight = images[0][0].height;

  //imageHeight = originImageHeight * imageWidth / originImageWidth;
  //imageWidth = originImageWidth * imageHeight / originImageHeight;

  canvasWidth = (imageWidth+ margin) * (imageNum-1) + imageWidth;
  canvasHeight = imageHeight;

  surface.setSize(canvasWidth, canvasHeight);

  noLoop();
}

void draw() {
  background(255);
  for (int j = 0; j < a; j++) {
    for (int i = 0; i < imageNum; i++) {
      image(images[j][1], (imageWidth+margin) * 0, 0, imageWidth, imageHeight);
      image(images[j][0], (imageWidth+margin) * 1, 0, imageWidth, imageHeight);
    }
    save( j+".jpg");
  }
  exit();
}
