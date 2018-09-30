package com.example.eglnn.layer;

import android.content.Context;
import android.text.TextUtils;


import com.example.eglnn.Render;
import com.example.eglnn.utils.ShaderUtils;
import com.example.eglnn.utils.TestDataCreator;
import com.example.eglnn.utils.Utils;

import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

import static com.example.eglnn.Render.initCompPro;
import static com.example.eglnn.utils.Constants.S_FULL_CONN_SHADER_HEADER;


/**
 * 全连接层 视为 1,1,amount的特殊image
 * 实际存储为一张texture w h c 为  ceil(amount/4), 1, 4
 */
public class FullConnSSBO extends Layer {

    private static final String TAG = "FullConnSSBO";

    private List<float[]> mKennels;
    private int mShaderPro;
    private int mKennelAmount;
    private String mParamFilePath;

    private int[] mParams;

    private ActiveType mType;
    private int[] mKennelBuffer = new int[1];
    private int mNumGroupZ;
    private boolean mPreLayerIsArray;

    public FullConnSSBO(Context context, String name, Layer preLayer, int kennelAmount, ActiveType type, String paramFilePath) {
        super(context, name, preLayer);
        this.mKennelAmount = kennelAmount;
        this.mType = type;
        this.mOutShape = calculateFullShape(kennelAmount);
        this.mParamFilePath = paramFilePath;
        if (mPreLayer instanceof Input || mPreLayer instanceof Pooling || mPreLayer instanceof ConvGEMM) {
            mPreLayerIsArray = true;
        }
    }

    private int[] calculateFullShape(int kennelAmount) {
        return new int[]{1, 1, kennelAmount};
    }

    @Override
    public void initialize() {
        initFullConnect();
    }

    private void initFullConnect() {
        int[] inputShape = mPreLayer.getOutputShape();
        int kennelSize = inputShape[0] * inputShape[1] * inputShape[2] + 1;
        int alignKennelSize =
                inputShape[0] * inputShape[1] * Utils.alignBy4(inputShape[2]) + 4;

        int xSize;
        if (Utils.alignBy4(mKennelAmount) / 4 <= 1024) {
            xSize = Utils.alignBy4(mKennelAmount) / 4;
            mNumGroupZ = 1;
        } else {
            xSize = 1024;
            mNumGroupZ = (int) Math.ceil(Utils.alignBy4(mKennelAmount) / 4 / 1024);
        }

        String source = createShaderSource(xSize);
        mShaderPro = initCompPro(source);

        mAttachID = Layer.getDataAttachID();
        mOutTex = Render.createTexture(Utils.alignBy4(mKennelAmount) / 4, 1);


        int kennelBufSize = alignKennelSize * mKennelAmount;
        if (TextUtils.isEmpty(mParamFilePath)) {
            mKennels = createKennels(alignKennelSize);
        } else {
            mKennels = loadKennels(kennelSize, alignKennelSize);
        }

        mKennelBuffer[0] = Render.initKennelBuffer(kennelBufSize);
        transferKennelToBuffer();


        createShaderParams();
    }

    private String createShaderSource(int xSize) {
        String shaderFile;
        if (mPreLayerIsArray) {
            shaderFile = "full_connect_array.comp";
        } else {
            shaderFile = "full_connect.comp";
        }
        String source = ShaderUtils.loadFromAssetsFile(shaderFile, mContext.getResources());
        return String.format(Locale.getDefault(), S_FULL_CONN_SHADER_HEADER, 1, mKennelAmount, xSize, 1, 1) + source;

    }

    private void createShaderParams() {
        mParams = new int[8];
        mParams[0] = mInShape[0];
        mParams[1] = mInShape[1];
        mParams[2] = mInShape[2];
        mParams[3] = mOutShape[0];
        mParams[4] = mOutShape[1];
        mParams[5] = mOutShape[2];
        mParams[6] = Utils.alignBy4(mInShape[2]) / 4;
        mParams[7] = mType.index;
    }

    private List<float[]> loadKennels(int kennelSize, int alignKennelSize) {
        return null;
    }

    private void transferKennelToBuffer() {
        for (int i = 0; i < mKennels.size(); i++) {
            float[] kennel = mKennels.get(i);
            int offset = i * kennel.length;
            Render.transferToBuffer(FloatBuffer.wrap(kennel), mKennelBuffer[0], offset);
        }
    }

    private List<float[]> createKennels(int alignSize) {
        List<float[]> kennels = new ArrayList<>();
        for (int i = 0; i < mKennelAmount; i++) {
            float[] kennel = TestDataCreator.createFullConnKennel(alignSize, mPreLayer.getOutputShape(), i);
            kennels.add(kennel);
        }
        return kennels;
    }

    @Override
    protected void bindTextureAndBuffer() {
        Render.bindTextureAndBuffer(mOutTex, mAttachID, mKennelBuffer[0]);
    }

    @Override
    protected void actualForwardProc(float[][] input) {
        Render.performFullConnectSSBO(mShaderPro, mParams, mPreLayer.getOutTex(), mOutTex, mKennelBuffer[0], mNumGroupZ);
    }

}
