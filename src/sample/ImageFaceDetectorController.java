package sample;

import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.awt.image.WritableRaster;
import java.io.*;
import java.nio.IntBuffer;

import static org.bytedeco.javacpp.opencv_core.CV_32SC1;
import static org.bytedeco.javacpp.opencv_core.CV_8UC3;
import static org.bytedeco.javacpp.opencv_imgcodecs.CV_LOAD_IMAGE_GRAYSCALE;
import static org.bytedeco.javacpp.opencv_imgcodecs.cvLoadImage;
import static org.bytedeco.javacpp.opencv_imgcodecs.imread;
import static org.bytedeco.javacpp.opencv_face.createFisherFaceRecognizer;
import static org.bytedeco.javacpp.opencv_face.createEigenFaceRecognizer;
import static org.bytedeco.javacpp.opencv_face.createLBPHFaceRecognizer;

import javafx.fxml.FXML;
import javafx.scene.control.TextField;
import javafx.scene.image.ImageView;
import javafx.scene.image.PixelWriter;
import javafx.scene.image.WritableImage;
import org.bytedeco.javacpp.opencv_face.FaceRecognizer;
import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_core.MatVector;

import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.Java2DFrameConverter;
import org.bytedeco.javacv.OpenCVFrameConverter;

import javax.imageio.ImageIO;

import static org.bytedeco.javacpp.opencv_core.*;
import static org.bytedeco.javacpp.opencv_highgui.*;
import static org.bytedeco.javacpp.opencv_imgproc.*;
import static org.bytedeco.javacpp.opencv_objdetect.*;

/**
 * Created by Eias on 16-Sep-16.
 */
public class ImageFaceDetectorController {
    @FXML
    private ImageView imageView;

    @FXML
    private TextField fileNameText;

    private MatVector images;
    private final String TRAINING_DATA = "resources/training.xml";

    @FXML
    protected void recognizeFaces() {
        Mat imageMat = imread("resources/" + fileNameText.getText());
        CascadeClassifier face_cascade = new CascadeClassifier("resources/haarcascade_frontalface_default.xml");
        Mat videoMatGray = new Mat();
        // Convert the current frame to grayscale:
        cvtColor(imageMat, videoMatGray, COLOR_BGRA2GRAY);
        equalizeHist(videoMatGray, videoMatGray);

        //Point p = new Point();
        RectVector faces = new RectVector();
        // Find the faces in the frame:
        face_cascade.detectMultiScale(videoMatGray, faces);
        System.out.println("faces: " + faces.size());

        // At this point you have the position of the faces in faces. Now we'll get the faces, make a prediction and
        // annotate it in the video. Cool or what?
        for (int i = 0; i < faces.size(); i++) {
            Rect face_i = faces.get(i);

            Mat face = new Mat(videoMatGray, face_i);
            // If fisher face recognizer is used, the face need to be resized.
            // resize(face, face_resized, new Size(im_width, im_height), 1.0, 1.0, INTER_CUBIC);

            // Now perform the prediction, see how easy that is:
            FaceRecognizer faceRecognizer = createLBPHFaceRecognizer();
            faceRecognizer.load(TRAINING_DATA);
            int prediction = faceRecognizer.predict(face);

            // And finally write all we've found out to the original image!
            // First of all draw a green rectangle around the detected face:
            rectangle(imageMat, face_i, new Scalar(0, 255, 0, 1));

            // Create the text we will annotate the box with:
            String box_text = "Prediction = " + prediction;
            // Calculate the position for annotated text (make sure we don't
            // put illegal values in there):
            int pos_x = Math.max(face_i.tl().x() - 10, 0);
            int pos_y = Math.max(face_i.tl().y() - 10, 0);
            // And now put it into the image:
            putText(imageMat, box_text, new Point(pos_x, pos_y), FONT_HERSHEY_PLAIN, 1.0, new Scalar(0, 255, 0, 2.0));
        }
        // Show the result:
        imshow("face_recognizer", imageMat);

        cvWaitKey(0);
    }

    @FXML
    protected void detectFaces() {
        IplImage img = cvLoadImage("resources/" + fileNameText.getText());
        CvHaarClassifierCascade cascade = new
                CvHaarClassifierCascade(cvLoad("resources/haarcascade_frontalface_default.xml"));
        CvMemStorage storage = CvMemStorage.create();

        CvSeq sign = cvHaarDetectObjects(
                img,
                cascade,
                storage,
                1.5,
                3,
                CV_HAAR_DO_CANNY_PRUNING);

        cvClearMemStorage(storage);

        int total_Faces = sign.total();

        for (int i = 0; i < total_Faces; i++) {
            CvRect r = new CvRect(cvGetSeqElem(sign, i));
            cvRectangle(
                    img,
                    cvPoint(r.x(), r.y()),
                    cvPoint(r.width() + r.x(), r.height() + r.y()),
                    CvScalar.RED,
                    2,
                    CV_AA,
                    0);

        }

        BufferedImage bf = IplImageToBufferedImage(img);

        WritableImage wr = null;
        if (bf != null) {
            wr = new WritableImage(bf.getWidth(), bf.getHeight());
            PixelWriter pw = wr.getPixelWriter();
            for (int x = 0; x < bf.getWidth(); x++) {
                for (int y = 0; y < bf.getHeight(); y++) {
                    pw.setArgb(x, y, bf.getRGB(x, y));
                }
            }
        }
        imageView.setImage(wr);
        //cvShowImage("Result", img);
    }

    @FXML
    protected void startTraining(){
        String trainingDir = "resources/training";
        //Mat testImage = imread("resources/test.jpg", CV_LOAD_IMAGE_GRAYSCALE);

        File root = new File(trainingDir);

        FilenameFilter imgFilter = new FilenameFilter() {
            public boolean accept(File dir, String name) {
                name = name.toLowerCase();
                return name.endsWith(".jpg") || name.endsWith(".pgm") || name.endsWith(".png");
            }
        };

        File[] imageFiles = root.listFiles(imgFilter);

        images = new MatVector(imageFiles.length);

        Mat labels = new Mat(imageFiles.length, 1, CV_32SC1);
        IntBuffer labelsBuf = labels.createBuffer();

        int counter = 0;

        for (File image : imageFiles) {
            Mat img = imread(image.getAbsolutePath(), CV_LOAD_IMAGE_GRAYSCALE);

            int label = Integer.parseInt(image.getName().split("\\_")[0]);

            images.put(counter, img);

            labelsBuf.put(counter, label);

            counter++;
        }

        //FaceRecognizer faceRecognizer = createFisherFaceRecognizer();
        // FaceRecognizer faceRecognizer = createEigenFaceRecognizer();
        FaceRecognizer faceRecognizer = createLBPHFaceRecognizer();

        faceRecognizer.train(images, labels);
        faceRecognizer.save(TRAINING_DATA);

        //int predictedLabel = faceRecognizer.predict(testImage);

        //System.out.println("Predicted label: " + predictedLabel);
    }

    public static BufferedImage IplImageToBufferedImage(IplImage src) {
        OpenCVFrameConverter.ToIplImage grabberConverter = new OpenCVFrameConverter.ToIplImage();
        Java2DFrameConverter paintConverter = new Java2DFrameConverter();
        Frame frame = grabberConverter.convert(src);
        return paintConverter.getBufferedImage(frame,1);
    }

    public static Mat bufferedImageToMat(BufferedImage bi) {
        Mat mat = new Mat(bi.getHeight(), bi.getWidth(), CV_8UC3);
        byte[] data = ((DataBufferByte) bi.getRaster().getDataBuffer()).getData();
        mat.arrayData().put(data);//.put(0, 0, data);
        return mat;
    }

    public static BufferedImage matToBufferedImage(Mat mat) {

        if (mat.rows() > 0 && mat.cols() > 0) {
            BufferedImage image = new BufferedImage(mat.cols(), mat.rows(), BufferedImage.TYPE_3BYTE_BGR);
            WritableRaster raster = image.getRaster();
            DataBufferByte dataBuffer = (DataBufferByte) raster.getDataBuffer();
            byte[] data = dataBuffer.getData();
            // convert byte array back to BufferedImage
            InputStream in = new ByteArrayInputStream(data);
            try {
                return ImageIO.read(in);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        return null;
    }
}
