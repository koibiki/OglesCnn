package com.example.oglescnn;

import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.view.View;

import com.example.cnnlib.CnnNetwork;
import com.example.cnnlib.layer.InputLayer;
import com.example.cnnlib.layer.NonLinearLayer;
import com.example.cnnlib.model.Kennel;
import com.example.cnnlib.model.LayerParams;

public class MainActivity extends AppCompatActivity {

    private CnnNetwork mCnnNetwork;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
    }

    @Override
    protected void onStart() {
        super.onStart();
        mCnnNetwork = new CnnNetwork();
        LayerParams inLayerParams = new LayerParams(64, 64, 4);
        mCnnNetwork.addLayer(new InputLayer(this, inLayerParams));


//        LayerParams convParams1 = new LayerParams(64, 64, 4, 64, 64, 64);
//        List<Kennel> kennels = new ArrayList<>();
//        for (int i = 0; i < 64; i++) {
//            kennels.add(createKennel(i + 1, 3, 4, 1));
//        }
//        mCnnNetwork.addLayer(new ConvolutionLayer(this, convParams1, kennels, 1, new int[]{1, 1}));
//
//
        LayerParams reluParams = new LayerParams(64, 64, 4);
        mCnnNetwork.addLayer(new NonLinearLayer(this, reluParams, NonLinearLayer.NonLinearType.TANH));
//
//
//        LayerParams convParams2 = new LayerParams(64, 64, 64, 64, 64, 200);
//        List<Kennel> kennels2 = new ArrayList<>();
//        for (int i = 0; i < 200; i++) {
//            kennels2.add(createKennel(i + 1, 3, 64, 1));
//        }
//        mCnnNetwork.addLayer(new ConvolutionLayer(this, convParams2, kennels2, 1, new int[]{1, 1}));

    }

    private Kennel createKennel(int num, int length, int channel, float bias) {
        float[] data = new float[length * length * channel + 1];
        for (int i = 0; i < length * length * channel; i++) {
            data[i] = 1f;
        }
        data[length * length * channel] = bias;
        return new Kennel(length, length, channel, data);
    }

    public void performCNN(View view) {
        mCnnNetwork.run();
    }
}
