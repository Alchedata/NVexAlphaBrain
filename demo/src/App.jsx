import { useState } from 'react';
import Sidebar from './components/Sidebar';
import TopBar from './components/TopBar';
import Home from './pages/Home';
import ProjectOverview from './pages/ProjectOverview';
import FailureMap from './pages/FailureMap';
import PatchPlan from './pages/PatchPlan';
import IterationRunner from './pages/IterationRunner';
import ImprovementReport from './pages/ImprovementReport';
import PlatformMemory from './pages/PlatformMemory';

const PAGES = {
  home:     Home,
  overview: ProjectOverview,
  failure:  FailureMap,
  patch:    PatchPlan,
  runner:   IterationRunner,
  report:   ImprovementReport,
  memory:   PlatformMemory,
};

export default function App() {
  const [page, setPage] = useState('home');
  const PageComponent = PAGES[page] || Home;

  return (
    <>
      <div className="bg-grid" />
      <div className="bg-glow-1" />
      <div className="bg-glow-2" />
      <div className="app-shell">
        <Sidebar active={page} onNav={setPage} />
        <div className="main-content">
          <TopBar page={page} onNav={setPage} />
          <PageComponent onNav={setPage} />
        </div>
      </div>
    </>
  );
}
