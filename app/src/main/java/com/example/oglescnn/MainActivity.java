package com.example.oglescnn;

import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.support.v7.app.AppCompatActivity;
import android.view.View;

import com.example.cnnlib.CnnNetwork;
import com.example.cnnlib.layer.ConvolutionLayer;
import com.example.cnnlib.layer.FlatLayer;
import com.example.cnnlib.layer.FullConnectLayer;
import com.example.cnnlib.layer.InputLayer;
import com.example.cnnlib.layer.Layer;
import com.example.cnnlib.layer.NonLinearLayer;
import com.example.cnnlib.layer.PoolingLayer;
import com.example.cnnlib.layer.SoftMaxLayer;

public class MainActivity extends AppCompatActivity {

    private CnnNetwork mCnnNetwork;
    private HandlerThread glThread;
    private Handler handler;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        glThread = new HandlerThread("GLThread");
        glThread.start();

        handler = new Handler(glThread.getLooper());
        handler.post(new Runnable() {
            @Override
            public void run() {
                buildNet();
            }
        });
    }

    @Override
    protected void onStart() {
        super.onStart();
    }

    private void buildNet() {
        mCnnNetwork = new CnnNetwork();
        try {
            Layer in = new InputLayer(this, new int[]{32, 32, 4});
            mCnnNetwork.addLayer(in);

            Layer conv1 = new ConvolutionLayer(this, in, new int[]{32, 32, 64}, new int[]{3, 3, 4}, 1, new int[]{1, 1});
            mCnnNetwork.addLayer(conv1);

            Layer relu1 = new NonLinearLayer(this, conv1, new int[]{32, 32, 64}, NonLinearLayer.NonLinearType.RELU);
            mCnnNetwork.addLayer(relu1);

            Layer pool1 = new PoolingLayer(this, conv1, new int[]{16, 16, 64}, new int[]{2, 2}, new int[]{2, 2});
            mCnnNetwork.addLayer(pool1);

            Layer conv2 = new ConvolutionLayer(this, pool1, new int[]{16, 16, 200}, new int[]{3, 3, 64}, 1, new int[]{1, 1});
            mCnnNetwork.addLayer(conv2);

            Layer pool2 = new PoolingLayer(this, conv2, new int[]{8, 8, 200}, new int[]{2, 2}, new int[]{2, 2});
            mCnnNetwork.addLayer(pool2);

            Layer flatLayer = new FlatLayer(this, pool2, new int[]{64, 200, 1});
            mCnnNetwork.addLayer(flatLayer);

            Layer fullConnLayer = new FullConnectLayer(this, flatLayer, new int[]{20, 1, 1});
            mCnnNetwork.addLayer(fullConnLayer);

//            Layer softmaxLayer = new SoftMaxLayer(this, fullConnLayer, new int[]{20, 1, 1});
//            mCnnNetwork.addLayer(softmaxLayer);

            mCnnNetwork.run();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }


    public void performCNN(View view) {
        handler.post(new Runnable() {
            @Override
            public void run() {
                mCnnNetwork.run();
            }
        });
    }

    public void showOutput(View view) {
        handler.post(new Runnable() {
            @Override
            public void run() {
                mCnnNetwork.readOutput();
            }
        });
    }

}
