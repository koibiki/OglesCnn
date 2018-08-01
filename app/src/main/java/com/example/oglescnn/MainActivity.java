package com.example.oglescnn;

import android.os.Bundle;
import android.os.Environment;
import android.support.v7.app.AppCompatActivity;
import android.view.View;

import com.example.cnnlib.CnnNetwork;
import com.example.cnnlib.layer.ConvSSBO;
import com.example.cnnlib.layer.Input;
import com.example.cnnlib.layer.Layer;
import com.example.cnnlib.layer.NonLinear;
import com.example.cnnlib.layer.Pooling;
import com.example.cnnlib.utils.DataUtils;

public class MainActivity extends AppCompatActivity {

    private CnnNetwork mCnnNetwork;
    private TestUtils testUtils;
    private String sRoot = Environment.getExternalStorageDirectory().getAbsolutePath();
    private String TAG = "MainActivity";


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        testUtils = new TestUtils();


//        buildNet();
//        buildLetNet();
//        buildCifaNet();
        buildTestNet();
    }

    private void buildTestNet() {
        mCnnNetwork = new CnnNetwork(this);

        Layer in = new Input(this, new int[]{32, 32, 3});
        mCnnNetwork.addLayer(in);

        Layer conv1 = new ConvSSBO(this, in, 6, new int[]{5, 5, 3}, 2, new int[]{1, 1}, NonLinear.NonLinearType.RELU, "");
        mCnnNetwork.addLayer(conv1);

        Layer pool1 = new Pooling(this, conv1, new int[]{2, 2}, new int[]{2, 2});
        mCnnNetwork.addLayer(pool1);

        Layer conv2 = new ConvSSBO(this, pool1, 16, new int[]{5, 5, 6}, 2, new int[]{1, 1}, NonLinear.NonLinearType.RELU, "");
        mCnnNetwork.addLayer(conv2);

        mCnnNetwork.initialize();
    }


    private void buildLetNet() {
//        mCnnNetwork = new CnnNetwork();
//
//        Layer in = new Input(this, new int[]{32, 32, 1});
//        mCnnNetwork.addLayer(in);
//
//        Layer conv1 = new ConvSSBO(this, in, 6, new int[]{5, 5, 1}, 0, new int[]{1, 1}, NonLinear.NonLinearType.RELU, "");
//        mCnnNetwork.addLayer(conv1);
//
//        Layer pool1 = new Pooling(this, conv1, new int[]{2, 2}, new int[]{2, 2});
//        mCnnNetwork.addLayer(pool1);
//
//        Layer conv2 = new ConvSSBO(this, pool1, 16, new int[]{5, 5, 6}, 0, new int[]{1, 1}, NonLinear.NonLinearType.RELU, "");
//        mCnnNetwork.addLayer(conv2);
//
//        Layer pool2 = new Pooling(this, conv2, new int[]{2, 2}, new int[]{2, 2});
//        mCnnNetwork.addLayer(pool2);
//
//        Layer flatLayer = new Flat(this, pool2);
//        mCnnNetwork.addLayer(flatLayer);
//
//        Layer full1 = new FullConnSSBO(this, flatLayer, new int[]{120, 1, 1}, NonLinear.NonLinearType.RELU);
//        mCnnNetwork.addLayer(full1);
//
//        Layer full2 = new FullConnSSBO(this, full1, new int[]{10, 1, 1}, NonLinear.NonLinearType.RELU);
//        mCnnNetwork.addLayer(full2);
//
//        Layer softmaxLayer = new SoftMax(this, full2, new int[]{10, 1, 1});
//        mCnnNetwork.addLayer(softmaxLayer);

//        mCnnNetwork.initialize();
    }

    private void buildNet() {
        try {
            mCnnNetwork = new CnnNetwork(this, sRoot + "/Cifar10es/net/Cifar10_def_test.txt");
            mCnnNetwork.initialize();
        } catch (Exception e) {
            e.printStackTrace();
        }


    }

    private void buildCifaNet() {
//        mCnnNetwork = new CnnNetwork();
//
//        Layer in = new Input(this, new int[]{32, 32, 3});
//        mCnnNetwork.addLayer(in);
//
//
//        Layer conv1 = new ConvSSBO(this, in, 32, new int[]{5, 5, 3}, 2, new int[]{1, 1}, NonLinear.NonLinearType.RELU, "");
//        mCnnNetwork.addLayer(conv1);
//
//        Layer pool1 = new Pooling(this, conv1, new int[]{2, 2}, new int[]{2, 2});
//        mCnnNetwork.addLayer(pool1);
//
//        Layer conv2 = new ConvSSBO(this, pool1, 32, new int[]{5, 5, 32}, 2, new int[]{1, 1}, NonLinear.NonLinearType.RELU, "");
//        mCnnNetwork.addLayer(conv2);
//
//        Layer pool2 = new Pooling(this, conv2, new int[]{2, 2}, new int[]{2, 2});
//        mCnnNetwork.addLayer(pool2);
//
//        Layer conv3 = new ConvSSBO(this, pool2, 64, new int[]{5, 5, 32}, 2, new int[]{1, 1}, NonLinear.NonLinearType.RELU, "");
//        mCnnNetwork.addLayer(conv2);
//
//        Layer pool3 = new Pooling(this, conv3, new int[]{2, 2}, new int[]{2, 2});
//        mCnnNetwork.addLayer(pool3);
//
//        Layer flatLayer = new Flat(this, pool3);
//        mCnnNetwork.addLayer(flatLayer);
//
//        Layer full1 = new FullConnSSBO(this, flatLayer, 64, NonLinear.NonLinearType.RELU, "");
//        mCnnNetwork.addLayer(full1);
//
//        Layer full2 = new FullConnSSBO(this, full1, 10, NonLinear.NonLinearType.RELU, "");
//        mCnnNetwork.addLayer(full2);
//
//        Layer full3 = new FullConnSSBO(this, full2, 10, NonLinear.NonLinearType.RELU, "");
//        mCnnNetwork.addLayer(full3);
//
//
//        Layer softmaxLayer = new SoftMax(this, full3, 10);
//        mCnnNetwork.addLayer(softmaxLayer);
//
//        mCnnNetwork.initialize();

    }

    public void performCNN(View view) {
        float[][][] input = DataUtils.createInputBuffer(new int[]{32, 32, 3});
//        float[][][] input = testUtils.getTestImage(this);
        mCnnNetwork.predict(input);

//        mCnnNetwork.run();
    }

    public void showOutput(View view) {
        mCnnNetwork.readOutput();
    }
}
