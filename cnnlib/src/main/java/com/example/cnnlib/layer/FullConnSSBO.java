package com.example.cnnlib.layer;

import android.content.Context;
import android.text.TextUtils;

import com.example.cnnlib.render.Render;
import com.example.cnnlib.utils.DataUtils;
import com.example.cnnlib.utils.NetUtils;
import com.example.cnnlib.utils.ParamUnpacker;

import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.List;

import static com.example.cnnlib.render.Render.initFullConnPro;

/**
 * 全连接前接 flat 或 全连接层 kennel 个数不超过4096
 * 全连接层的 输入 输出 的channel 和 kennel 都需要 4 对齐
 */
public class FullConnSSBO extends Layer {

    private static final String TAG = "FullConnSSBO";

    private List<float[]> mKennels;
    private int mShaderPro;
    private int mKennelAmount;
    private String mParamFilePath;

    private int[] mParams;

    private NonLinear.Type mType;
    private int[] mKennelBuffer = new int[1];

    public FullConnSSBO(Context context, Layer preLayer, int kennelAmount, NonLinear.Type type, String paramFilePath) {
        super(context, preLayer);
        this.mKennelAmount = kennelAmount;
        this.mType = type;
        this.mOutputShape = calculateFullShape(kennelAmount);
        this.mParamFilePath = paramFilePath;
    }

    private int[] calculateFullShape(int kennelAmount) {
        int width = (int) Math.ceil(kennelAmount * 1.0f / 4);
        int height = 1;
        int channel = 4;
        return new int[]{width, height, channel};
    }

    @Override
    public void initialize() {
        initFullConnect();
    }

    private void initFullConnect() {
        int[] inputShape = mPreLayer.getOutputShape();
        int kennelSize = inputShape[0] * inputShape[1] * inputShape[2];
        int alignKennelSize =
                inputShape[0] * inputShape[1] * NetUtils.alignBy4(inputShape[2]) + 4;
        mShaderPro = initFullConnPro(mContext, "full_connect.comp", alignKennelSize, mKennelAmount, mOutputShape[0], 1, 1);
        mAttachID = Layer.getDataAttachID();
        mOutTex = Render.createTexture();


        int kennelBufSize = alignKennelSize * mKennelAmount;
        if (TextUtils.isEmpty(mParamFilePath)) {
            mKennels = createKennels(alignKennelSize);
        } else {
            mKennels = loadKennels(kennelSize, alignKennelSize);
        }

        mKennelBuffer[0] = Render.initKennelBuffer(kennelBufSize);
        transferKennelToBuffer();

        mParams = new int[7];
        mParams[0] = inputShape[0];
        mParams[1] = inputShape[1];
        mParams[2] = inputShape[2];
        mParams[3] = mOutputShape[0];
        mParams[4] = mOutputShape[1];
        mParams[5] = mOutputShape[2];
        mParams[6] = mType.index;
    }

    private List<float[]> loadKennels(int kennelSize, int alignKennelSize) {
        ParamUnpacker paramUnpacker = new ParamUnpacker();
        Object[] objects = paramUnpacker.unpackerFunction(mParamFilePath, new Class[]{float[].class, float[].class});
        float[] weight = (float[]) objects[0];
        float[] bias = (float[]) objects[1];

        int wSize = kennelSize - 1;

        List<float[]> kennels = new ArrayList<>();
        for (int i = 0; i < mKennelAmount; i++) {
            float[] kennel = new float[alignKennelSize];
            for (int s = 0; s < wSize; s++) {
                int wIndex = i * wSize + s;
                kennel[s] = weight[wIndex];
            }
            kennel[alignKennelSize - 4] = bias[i];
            kennels.add(kennel);
        }
        return kennels;
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
            float[] kennel = DataUtils.createFullConnKennel(alignSize, mPreLayer.getOutputShape(), i);
            kennels.add(kennel);
        }
        return kennels;
    }

    @Override
    protected void bindTextureAndBuffer() {
        Render.bindTextureAndBuffer(mOutTex, mAttachID, mKennelBuffer[0]);
    }

    @Override
    protected void actualForwardProc(float[][][] input) {
        Render.performFullConnectSSBO(mShaderPro, mParams, mPreLayer.getOutTex(), mOutTex, mKennelBuffer[0]);
    }

}
