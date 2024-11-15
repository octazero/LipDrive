import UIKit
import AVFoundation
import RecordButton


class CameraViewController: UIViewController,AVCaptureVideoDataOutputSampleBufferDelegate, AVCaptureMetadataOutputObjectsDelegate {
    let MainColor = UIColor(red: 96/255, green: 195/255,blue: 157/255, alpha: 1.0)
    let statusBar = UIApplication.shared.value(forKeyPath: "statusBarWindow.statusBar") as? UIView
    
    @IBOutlet weak var closeButton: UIButton!
    @IBOutlet weak var cameraNavigationBar: UINavigationBar!
    
    var recordButton : RecordButton!
    var progressTimer : Timer!
    var progress : CGFloat! = 0
    let maxProgressDuration = 10
    let timeInterval = 0.1
    
    var isPressed = false
    
    var startTime = 0
    var endTime = 0
    var TotalFrames = 0
    
    
    var session = AVCaptureSession()
    let layer = AVSampleBufferDisplayLayer()
    let sampleQueue = DispatchQueue(label: "com.MIU.LipDrive.sampleQueue", attributes: [])
    let faceQueue = DispatchQueue(label: "com.MIU.LipDrive.faceQueue", attributes: [])
    let wrapper = DlibWrapper()
    
    var currentMetadata: [AnyObject] = []
    var ArrayOfPoints = [CGPoint]()
    
    var checkResults = false
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        currentMetadata = []
        
        self.statusBar?.backgroundColor = self.MainColor
        
        self.recordButton = RecordButton(frame: CGRect(x: 157.5, y: 587, width: 60, height: 60))
        
        self.recordButton.addTarget(self, action: #selector(isRecording), for: .touchDown)
        self.recordButton.addTarget(self, action: #selector(didFinishRecording), for: UIControlEvents.touchUpInside)
    }
    
    override func viewDidAppear(_ animated: Bool) {
        super.viewDidAppear(animated)
        
        self.openSession()
        
        let layer = self.layer
        layer.frame = self.view.bounds
        
        self.view.layer.addSublayer(layer)
        self.view.addSubview(self.recordButton)
        self.view.addSubview(self.cameraNavigationBar)
        self.view.addSubview(self.closeButton)
        
        view.layoutIfNeeded()
        
    }
    
    @IBAction func CloseCameraFrame(_ sender: UIButton) {
        self.dismiss(animated: true, completion: nil)
    }
    
    @objc func isRecording() {
        self.ArrayOfPoints.removeAll()
        self.isPressed = true
        self.checkResults = false
        
        self.progress = 0
        self.TotalFrames = 0
        
        self.progressTimer = Timer.scheduledTimer(timeInterval: self.timeInterval, target: self, selector: #selector(isStillRecording), userInfo: nil, repeats: true)
        
        let date = Date()
        self.startTime = Calendar.current.component(.second, from: date)
    }
    
    @objc func isStillRecording() {
        let maxDuration = CGFloat(self.maxProgressDuration) // max duration of the recordButton
        
        progress = progress + (CGFloat(self.timeInterval) / maxDuration)
        self.recordButton.setProgress(progress)
        
        if progress >= 1 {
            progressTimer.invalidate()
            self.recordButton.didTouchUp()
        }
        self.isPressed = true
    }
    
    
    @objc func didFinishRecording() {
        self.progressTimer.invalidate()
        self.isPressed = false
        
        let date = Date()
        self.endTime = Calendar.current.component(.second, from: date)
        
        if self.checkResults == false{
            self.sendAndReceiveData()
        }
    }
    
    func sendAndReceiveData(){
        let urlString = "http://192.168.6.1:8000/"
        let url = URL(string: urlString)
        print("\n\nURL \(url?.absoluteString)")
        
        URLSession.shared.dataTask(with: url!) { (Data, Response, Error) in
            if Error != nil{
                //Error
                print("HERE!!!")
                print("Error Description: \(Error?.localizedDescription ?? "")")
            }else{
                //Success
                print(Response)
            }
        }.resume()
        print("\n\n")
    }
    
    
    
    func openSession() {
        let device = AVCaptureDevice.devices(for: AVMediaType.video)
            .map { $0 }
            .filter { $0.position == .front}
            .first!
        
        let input = try! AVCaptureDeviceInput(device: device)
        
        let output = AVCaptureVideoDataOutput()
        output.setSampleBufferDelegate(self, queue: sampleQueue)
        
        let metaOutput = AVCaptureMetadataOutput()
        metaOutput.setMetadataObjectsDelegate(self, queue: faceQueue)
        
        session.beginConfiguration()
        
        if session.canAddInput(input) {
            session.addInput(input)
        }
        if session.canAddOutput(output) {
            session.addOutput(output)
        }
        if session.canAddOutput(metaOutput) {
            session.addOutput(metaOutput)
        }
        
        session.commitConfiguration()
        
        let settings: [AnyHashable: Any] = [kCVPixelBufferPixelFormatTypeKey as AnyHashable: Int(kCVPixelFormatType_32BGRA)]
        output.videoSettings = settings as! [String : Any]
        
        metaOutput.metadataObjectTypes = [AVMetadataObject.ObjectType.face]
        
        wrapper?.prepare()
        
        session.startRunning()
    }
    
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        connection.videoOrientation = AVCaptureVideoOrientation.portrait
        connection.isVideoMirrored = true
        
        if !currentMetadata.isEmpty {
            let boundsArray = currentMetadata
                .compactMap { $0 as? AVMetadataFaceObject }
                .map { (faceObject) -> NSValue in
                    let convertedObject = output.transformedMetadataObject(for: faceObject, connection: connection)
                    return NSValue(cgRect: convertedObject!.bounds)
            }
            if self.ArrayOfPoints.count < 580{
                wrapper?.changeRecordingButtonBoolean(self.isPressed)
                let Value = wrapper?.doWork(on: sampleBuffer, inRects: boundsArray)
                if Value != nil{
                    self.TotalFrames += 1
                    //print("\nID = \(self.TotalFrames), Count = \(String(describing: Value?.count)), Value = \(String(describing: Value))")
                    for Point in Value!{
                        self.ArrayOfPoints.append(Point as! CGPoint)
                    }
                }
            }else{
                if self.checkResults == false{
                    self.progressTimer.invalidate()
                    self.isPressed = false
                    self.recordButton.didTouchUp()
                }
            }
        }
        
        layer.enqueue(sampleBuffer)
    }
    
    func captureOutput(_ output: AVCaptureOutput, didDrop sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        //print("DidDropSampleBuffer")
    }
    
    // MARK: AVCaptureMetadataOutputObjectsDelegate
    func metadataOutput(_ output: AVCaptureMetadataOutput, didOutput metadataObjects: [AVMetadataObject], from connection: AVCaptureConnection) {
        currentMetadata = metadataObjects as [AnyObject]
    }
}
