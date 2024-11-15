import UIKit

class HomeController: UIViewController,UICollectionViewDataSource,UICollectionViewDelegate,UICollectionViewDelegateFlowLayout{
    @IBOutlet weak var myCollectionView: UICollectionView!
    @IBOutlet weak var myCollectionView2: UICollectionView!
    @IBOutlet weak var NavigationBarItem: NavigationItemDesignable!
    
    
    private var defaultSelectedIndex = 1
    private var previousSelectedIndex = 1
    private var Tabs = ["History", "Home", "Settings"]
    
    let HomePage: HomePageViewController = HomePageViewController(nibName: "HomePageViewController", bundle: nil)
    let HistoryPage: HistoryPageViewController = HistoryPageViewController(nibName: "HistoryPageViewController", bundle: nil)
    let SettingsPage: SettingsPageViewController = SettingsPageViewController(nibName: "SettingsPageViewController", bundle: nil)
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        self.changeTextLabelValue()
        self.changeScrollViewPage(animated: false)
    }
    
    public func getUIView()->UIView{
        return self.view
    }
    
    func collectionView(_ collectionView: UICollectionView, numberOfItemsInSection section: Int) -> Int {
        return self.Tabs.count
    }
    
    func collectionView(_ collectionView: UICollectionView, cellForItemAt indexPath: IndexPath) -> UICollectionViewCell {
        if collectionView == self.myCollectionView{
            let cell = collectionView.dequeueReusableCell(withReuseIdentifier: "CollectionViewCell", for: indexPath as IndexPath) as! MenuBarCollectionViewCell
            cell.myImage.image = UIImage(named: Tabs[indexPath.row])
            cell.setupHorizontalBar()
            
            if indexPath.row != self.defaultSelectedIndex{
                cell.myImage.image = cell.myImage.image!.withRenderingMode(UIImageRenderingMode.alwaysTemplate)
                cell.myImage.tintColor = UIColor(red:114/255,green:144/255,blue:144/255,alpha:1.0)
                cell.removeHorizontalBar()
            }
            return cell
        }else if collectionView == self.myCollectionView2{
            let Cell = collectionView.dequeueReusableCell(withReuseIdentifier: "CellPages", for: indexPath as IndexPath) as! PagesCollectionViewCell
            switch indexPath.row{
                case 0:
                    Cell.PageUIView.addSubview(HistoryPage.view)
                case 1:
                    Cell.PageUIView.addSubview(HomePage.view)
                case 2:
                    Cell.PageUIView.addSubview(SettingsPage.view)
                default:
                    print("Invalid")
            }
                
            
            return Cell
        }
        
        let fakeCell = collectionView.dequeueReusableCell(withReuseIdentifier: "CellPages", for: indexPath as IndexPath) as! PagesCollectionViewCell
        return fakeCell
    }
    
    func collectionView(_ collectionView: UICollectionView,layout collectionViewLayout: UICollectionViewLayout,sizeForItemAt indexPath: IndexPath) -> CGSize{
        if collectionView == self.myCollectionView{
            return CGSize(width: self.myCollectionView.frame.width/3, height: self.myCollectionView.frame.height)
        }else if collectionView == self.myCollectionView2{
            return CGSize(width: self.myCollectionView2.frame.width, height: self.myCollectionView2.frame.height)
        }
        return CGSize(width: 0, height: 0)
    }
    
    func collectionView(_ collectionView: UICollectionView, layout collectionViewLayout: UICollectionViewLayout, minimumInteritemSpacingForSectionAt section: Int) -> CGFloat {
        return 0
    }
    
    func collectionView(_ collectionView: UICollectionView, didSelectItemAt indexPath: IndexPath) {
        // handle tap events
        if collectionView == self.myCollectionView{
            self.previousSelectedIndex = self.defaultSelectedIndex
            self.defaultSelectedIndex = indexPath.row
            self.changeTextLabelValue()
            self.myCollectionView.reloadData()
            self.changeScrollViewPage(animated: true)
        }
    }
    
    func scrollViewDidScroll(_ scrollView: UIScrollView) {
        self.previousSelectedIndex = self.defaultSelectedIndex
        self.defaultSelectedIndex = Int(scrollView.contentOffset.x / scrollView.frame.size.width)
        self.changeTextLabelValue()
        self.myCollectionView.reloadData()
    }
    
    func changeTextLabelValue(){
        self.NavigationBarItem?.titleLabelText = Tabs[self.defaultSelectedIndex]
    }
    
    func changeScrollViewPage(animated: Bool){
        var Direction: UICollectionViewScrollPosition!
        switch self.defaultSelectedIndex {
        case 0:
            Direction = UICollectionViewScrollPosition.left
        case 1:
            if self.previousSelectedIndex == 2{
                Direction = UICollectionViewScrollPosition.left
            }else if self.previousSelectedIndex == 0{
                Direction = UICollectionViewScrollPosition.right
            }else if self.previousSelectedIndex == 1{
                Direction = UICollectionViewScrollPosition.right
            }
        case 2:
            Direction = UICollectionViewScrollPosition.right
        default:
            print("Invalid")
        }
        
        self.myCollectionView2?.scrollToItem(at:IndexPath(item: self.defaultSelectedIndex, section: 0), at: Direction, animated: animated)
    }
}
